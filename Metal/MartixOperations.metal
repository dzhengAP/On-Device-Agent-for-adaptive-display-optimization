//
//  MetalShaders.metal
//  On Display Computing
//
//  Metal compute shaders for LLM acceleration
//

#include <metal_stdlib>
using namespace metal;

// MARK: - Matrix Operations

/// High-performance matrix multiplication
/// Performs C = A * B where:
/// - A is M x K matrix
/// - B is K x N matrix
/// - C is M x N result matrix
kernel void matrix_multiply(
    device const float* matrixA [[buffer(0)]],
    device const float* matrixB [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // Check bounds
    if (gid.x >= N || gid.y >= M) return;
    
    float sum = 0.0f;
    // Compute dot product for this position
    for (uint k = 0; k < K; k++) {
        sum += matrixA[gid.y * K + k] * matrixB[k * N + gid.x];
    }
    
    result[gid.y * N + gid.x] = sum;
}

/// Optimized matrix multiplication with shared memory
kernel void matrix_multiply_optimized(
    device const float* matrixA [[buffer(0)]],
    device const float* matrixB [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    // Tile size for optimization
    // Tile size for optimization
    constexpr uint TILE_SIZE = 16;

    // Shared memory tiles
    threadgroup float tileA[TILE_SIZE][TILE_SIZE];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE];

    
    // Check bounds
    uint row = tgid.y * TILE_SIZE + lid.y;
    uint col = tgid.x * TILE_SIZE + lid.x;
    
    float sum = 0.0f;
    
    // Process tiles
    for (uint t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile A
        uint aRow = row;
        uint aCol = t * TILE_SIZE + lid.x;
        tileA[lid.y][lid.x] = (aRow < M && aCol < K) ?
            matrixA[aRow * K + aCol] : 0.0f;
        
        // Load tile B
        uint bRow = t * TILE_SIZE + lid.y;
        uint bCol = col;
        tileB[lid.y][lid.x] = (bRow < K && bCol < N) ?
            matrixB[bRow * N + bCol] : 0.0f;
        
        // Synchronize threads
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial sum
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[lid.y][k] * tileB[k][lid.x];
        }
        
        // Synchronize before next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (row < M && col < N) {
        result[row * N + col] = sum;
    }
}

// MARK: - Activation Functions

/// GELU (Gaussian Error Linear Unit) activation function
/// GELU(x) = x * Φ(x) where Φ(x) is the cumulative distribution function
kernel void gelu_activation(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    float x = input[gid];
    
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    float x3 = x * x * x;
    float inner = sqrt(2.0f / M_PI_F) * (x + 0.044715f * x3);
    float tanh_val = tanh(inner);
    
    output[gid] = 0.5f * x * (1.0f + tanh_val);
}

/// ReLU activation function
kernel void relu_activation(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    output[gid] = max(0.0f, input[gid]);
}

/// Swish/SiLU activation function
kernel void swish_activation(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    float x = input[gid];
    float sigmoid = 1.0f / (1.0f + exp(-x));
    output[gid] = x * sigmoid;
}

// MARK: - Softmax Operations

/// Temperature-scaled softmax function
/// Applies temperature scaling then softmax normalization
kernel void softmax_temperature(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& vocab_size [[buffer(2)]],
    constant float& temperature [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= vocab_size) return;
    
    // Apply temperature scaling
    float scaled_logit = input[gid] / temperature;
    
    // Find max value for numerical stability (requires reduction)
    float max_val = scaled_logit;
    for (uint i = 0; i < vocab_size; i++) {
        max_val = max(max_val, input[i] / temperature);
    }
    
    // Compute exponential
    float exp_val = exp(scaled_logit - max_val);
    
    // Compute sum of all exponentials (requires reduction)
    float sum = 0.0f;
    for (uint i = 0; i < vocab_size; i++) {
        sum += exp((input[i] / temperature) - max_val);
    }
    
    // Final probability
    output[gid] = exp_val / sum;
}

/// Standard softmax function
kernel void softmax(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& vocab_size [[buffer(2)]],
    constant uint& stride [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= vocab_size) return;
    
    // Find maximum value for numerical stability
    float max_val = input[0];
    for (uint i = 0; i < vocab_size; i++) {
        max_val = max(max_val, input[i]);
    }
    
    // Compute exponential and sum
    float exp_val = exp(input[gid] - max_val);
    float sum = 0.0f;
    
    for (uint i = 0; i < vocab_size; i++) {
        sum += exp(input[i] - max_val);
    }
    
    // Final probability
    output[gid] = exp_val / sum;
}

// MARK: - Embedding Operations

/// Embedding lookup operation
/// Retrieves embedding vectors for given token IDs
kernel void embedding_lookup(
    device const float* embeddings [[buffer(0)]],    // [vocab_size, embed_dim]
    device const uint* token_ids [[buffer(1)]],      // [seq_len]
    device float* output [[buffer(2)]],               // [seq_len, embed_dim]
    constant uint& vocab_size [[buffer(3)]],
    constant uint& embed_dim [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]            // (seq_pos, embed_idx)
) {
    uint seq_pos = gid.x;
    uint embed_idx = gid.y;
    
    if (seq_pos >= seq_len || embed_idx >= embed_dim) return;
    
    uint token_id = token_ids[seq_pos];
    
    // Bounds check for token ID
    if (token_id >= vocab_size) return;
    
    // Copy embedding value
    output[seq_pos * embed_dim + embed_idx] =
        embeddings[token_id * embed_dim + embed_idx];
}

// MARK: - Attention Mechanisms

/// Multi-head attention computation
/// Computes scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
kernel void multihead_attention(
    device const float* queries [[buffer(0)]],       // [seq_len, d_model]
    device const float* keys [[buffer(1)]],          // [seq_len, d_model]
    device const float* values [[buffer(2)]],        // [seq_len, d_model]
    device float* output [[buffer(3)]],              // [seq_len, d_model]
    device float* attention_weights [[buffer(4)]],   // [seq_len, seq_len]
    constant uint& seq_len [[buffer(5)]],
    constant uint& d_model [[buffer(6)]],
    constant uint& num_heads [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]           // (head, seq_i, seq_j)
) {
    uint head = gid.x;
    uint seq_i = gid.y;
    uint seq_j = gid.z;
    
    if (head >= num_heads || seq_i >= seq_len || seq_j >= seq_len) return;
    
    uint head_dim = d_model / num_heads;
    float scale = 1.0f / sqrt(float(head_dim));
    
    // Compute attention score for this head
    float score = 0.0f;
    uint q_offset = seq_i * d_model + head * head_dim;
    uint k_offset = seq_j * d_model + head * head_dim;
    
    for (uint d = 0; d < head_dim; d++) {
        score += queries[q_offset + d] * keys[k_offset + d];
    }
    
    score *= scale;
    
    // Store raw attention score
    attention_weights[seq_i * seq_len + seq_j] = score;
    
    // Apply softmax across seq_j dimension (requires synchronization)
    // This is simplified - in practice would use threadgroup memory
}

// MARK: - Layer Normalization

/// Layer normalization
/// Normalizes across the feature dimension: (x - μ) / σ * γ + β
kernel void layer_norm(
    device const float* input [[buffer(0)]],         // [batch, seq_len, features]
    device const float* gamma [[buffer(1)]],         // [features] - scale parameter
    device const float* beta [[buffer(2)]],          // [features] - shift parameter
    device float* output [[buffer(3)]],              // [batch, seq_len, features]
    constant uint& features [[buffer(4)]],
    constant float& eps [[buffer(5)]],               // Small epsilon for stability
    uint2 gid [[thread_position_in_grid]]           // (batch*seq, feature)
) {
    uint batch_seq = gid.x;
    uint feature = gid.y;
    
    if (feature >= features) return;
    
    // Compute mean
    float sum = 0.0f;
    for (uint f = 0; f < features; f++) {
        sum += input[batch_seq * features + f];
    }
    float mean = sum / float(features);
    
    // Compute variance
    float var_sum = 0.0f;
    for (uint f = 0; f < features; f++) {
        float diff = input[batch_seq * features + f] - mean;
        var_sum += diff * diff;
    }
    float variance = var_sum / float(features);
    
    // Normalize and scale
    float normalized = (input[batch_seq * features + feature] - mean) / sqrt(variance + eps);
    output[batch_seq * features + feature] = normalized * gamma[feature] + beta[feature];
}

// MARK: - Utility Functions

/// Add bias to matrix
kernel void add_bias(
    device const float* input [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]           // (row, col)
) {
    if (gid.x >= cols || gid.y >= rows) return;
    
    uint index = gid.y * cols + gid.x;
    output[index] = input[index] + bias[gid.x];
}

/// Element-wise multiplication
kernel void elementwise_multiply(
    device const float* inputA [[buffer(0)]],
    device const float* inputB [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    output[gid] = inputA[gid] * inputB[gid];
}

/// Element-wise addition
kernel void elementwise_add(
    device const float* inputA [[buffer(0)]],
    device const float* inputB [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    output[gid] = inputA[gid] + inputB[gid];
}

// MARK: - Reduction Operations

/// Parallel reduction to find maximum value
kernel void reduce_max(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    // Shared memory for this threadgroup
    threadgroup float shared_data[256];
    
    // Load data into shared memory
    if (gid < count) {
        shared_data[lid] = input[gid];
    } else {
        shared_data[lid] = -INFINITY;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Perform reduction in shared memory
    for (uint s = group_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared_data[lid] = max(shared_data[lid], shared_data[lid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result for this threadgroup
    if (lid == 0) {
        output[gid / group_size] = shared_data[0];
    }
}

/// Parallel reduction to compute sum
kernel void reduce_sum(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    threadgroup float shared_data[256];
    
    // Load data
    if (gid < count) {
        shared_data[lid] = input[gid];
    } else {
        shared_data[lid] = 0.0f;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Reduction
    for (uint s = group_size / 2; s > 0; s >>= 1) {
        if (lid < s) {
            shared_data[lid] += shared_data[lid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (lid == 0) {
        output[gid / group_size] = shared_data[0];
    }
}

// MARK: - Specialized LLM Operations

/// Position encoding addition
/// Adds sinusoidal position encodings to input embeddings
kernel void add_position_encoding(
    device const float* input_embeddings [[buffer(0)]],    // [seq_len, d_model]
    device float* output [[buffer(1)]],                     // [seq_len, d_model]
    constant uint& seq_len [[buffer(2)]],
    constant uint& d_model [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]                  // (seq_pos, dim)
) {
    uint pos = gid.x;
    uint dim = gid.y;
    
    if (pos >= seq_len || dim >= d_model) return;
    
    float position = float(pos);
    float dimension = float(dim);
    
    // Compute position encoding
    float pos_enc;
    if (dim % 2 == 0) {
        // Even dimensions: sin
        pos_enc = sin(position / pow(10000.0f, dimension / float(d_model)));
    } else {
        // Odd dimensions: cos
        pos_enc = cos(position / pow(10000.0f, (dimension - 1) / float(d_model)));
    }
    
    // Add to input embedding
    uint index = pos * d_model + dim;
    output[index] = input_embeddings[index] + pos_enc;
}

/// Rotary Position Embedding (RoPE)
/// Applies rotary position embedding to query/key vectors
kernel void apply_rope(
    device const float* input [[buffer(0)]],               // [seq_len, num_heads, head_dim]
    device float* output [[buffer(1)]],                    // [seq_len, num_heads, head_dim]
    constant uint& seq_len [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]                 // (seq, head, dim_pair)
) {
    uint seq_pos = gid.x;
    uint head = gid.y;
    uint dim_pair = gid.z;
    
    if (seq_pos >= seq_len || head >= num_heads || dim_pair >= head_dim / 2) return;
    
    uint base_idx = seq_pos * num_heads * head_dim + head * head_dim + dim_pair * 2;
    
    float x = input[base_idx];
    float y = input[base_idx + 1];
    
    float position = float(seq_pos);
    float freq = 1.0f / pow(10000.0f, 2.0f * float(dim_pair) / float(head_dim));
    float angle = position * freq;
    
    float cos_angle = cos(angle);
    float sin_angle = sin(angle);
    
    // Apply rotation
    output[base_idx] = x * cos_angle - y * sin_angle;
    output[base_idx + 1] = x * sin_angle + y * cos_angle;
}

// MARK: - Quantization Support

/// Dequantize INT8 weights to FP32
kernel void dequantize_int8(
    device const char* quantized [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device const float* zeros [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    float scale = scales[gid / 128];  // Assuming 128 elements per scale
    float zero = zeros[gid / 128];
    
    output[gid] = (float(quantized[gid]) - zero) * scale;
}

/// Quantize FP32 to INT8
kernel void quantize_fp32_to_int8(
    device const float* input [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device const float* zeros [[buffer(2)]],
    device char* output [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    float scale = scales[gid / 128];
    float zero = zeros[gid / 128];
    
    float quantized = input[gid] / scale + zero;
    output[gid] = char(clamp(quantized, -128.0f, 127.0f));
}

// MARK: - Sampling Operations

/// Top-k sampling
/// Selects the top-k most probable tokens and samples from them
kernel void top_k_sampling(
    device const float* logits [[buffer(0)]],
    device float* probabilities [[buffer(1)]],
    device uint* top_k_indices [[buffer(2)]],
    constant uint& vocab_size [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    constant float& temperature [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= vocab_size) return;
    
    // Apply temperature scaling
    float scaled_logit = logits[gid] / temperature;
    
    // This is a simplified version - full top-k requires sorting
    // In practice, you'd use a more sophisticated approach
    probabilities[gid] = scaled_logit;
}

/// Nucleus (top-p) sampling preparation
kernel void nucleus_sampling_prep(
    device const float* probabilities [[buffer(0)]],
    device float* sorted_probs [[buffer(1)]],
    device uint* sorted_indices [[buffer(2)]],
    constant uint& vocab_size [[buffer(3)]],
    constant float& p_threshold [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= vocab_size) return;
    
    // This would need to be implemented with a proper sorting algorithm
    // For now, just copy the probabilities
    sorted_probs[gid] = probabilities[gid];
    sorted_indices[gid] = gid;
}

// MARK: - Memory Management Utilities

/// Clear buffer with specific value
kernel void clear_buffer(
    device float* buffer [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    constant float& value [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    buffer[gid] = value;
}

/// Copy buffer
kernel void copy_buffer(
    device const float* source [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    destination[gid] = source[gid];
}

/// Scale buffer by constant
kernel void scale_buffer(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    output[gid] = input[gid] * scale;
}
