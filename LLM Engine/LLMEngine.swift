//
//  LLMEngine.swift
//  On Display Computing
//
//  Created by David Zheng on 6/22/25.
//

import Foundation
import CoreML
import MetalPerformanceShaders
class LLMEngine {
    private var model: MLModel
    private var kvCache: [String: MLMultiArray]? = nil
    
    // This is the fixed sequence length required by your model
    private let requiredSequenceLength = 10
    
    init?() {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            
            // Choose ONE model to load
            let wrapper = try displayExpert(configuration: config) // or TinyLlama
            self.model = wrapper.model
            
            // Metal setup
            metalDevice = MTLCreateSystemDefaultDevice()
            if let device = metalDevice {
                commandQueue = device.makeCommandQueue()
                print("🟢 Metal device available: \(device.name)")
                
                do {
                    library = try device.makeDefaultLibrary(bundle: Bundle.main)
                    if loadMetalShaders() {
                        print("✅ Metal shaders loaded successfully")
                    }
                } catch {
                    print("⚠️ Failed to load Metal shaders: \(error)")
                }
            }
            
            print("🟢 Model loaded successfully")
        } catch {
            print("❌ Failed to load model: \(error)")
            return nil
        }
    }
    // MARK: - METAL
    // Metal properties
    private var metalDevice: MTLDevice?
    private var commandQueue: MTLCommandQueue?
    private var library: MTLLibrary?
    private var useMetalOptimization = false
    private var useMPS = false
    private var computeUnits: MLComputeUnits = .all

    // Pipeline states
    private var matrixMultiplyPipeline: MTLComputePipelineState?
    private var softmaxPipeline: MTLComputePipelineState?
    private var addBiasPipeline: MTLComputePipelineState?
    private var geluPipeline: MTLComputePipelineState?
    private var embeddingLookupPipeline: MTLComputePipelineState?

    // Metal functions
    private var matrixMultiplyFunc: MTLFunction?
    private var softmaxFunc: MTLFunction?
    private var addBiasFunc: MTLFunction?
    private var geluFunc: MTLFunction?
    private var embeddingLookupFunc: MTLFunction?

    /// Set compute units for the model
    func setComputeUnits(_ units: MLComputeUnits) {
        self.computeUnits = units
        
        // Reload model with new configuration
        do {
            let config = MLModelConfiguration()
            config.computeUnits = units
            
            // Use the auto-generated wrapper
            let wrapper = try displayExpert(configuration: config)
            self.model = wrapper.model
            
            print("🔄 Model reloaded with compute units: \(units)")
        } catch {
            print("❌ Failed to reload model: \(error)")
        }
    }

    /// Enable or disable Metal optimization
    func setMetalOptimizationEnabled(_ enabled: Bool) {
        useMetalOptimization = enabled
        
        if enabled && metalDevice == nil {
            print("⚠️ Metal optimization enabled but no Metal device available")
        } else if enabled {
            print("✅ Metal optimization enabled")
        } else {
            print("ℹ️ Metal optimization disabled")
        }
    }

    /// Enable or disable Metal Performance Shaders
    func setMPSEnabled(_ enabled: Bool) {
        useMPS = enabled
        
        if enabled && (metalDevice == nil || !metalDevice!.supportsFamily(.apple3)) {
            print("⚠️ MPS enabled but not supported on this device")
        } else if enabled {
            print("✅ MPS enabled")
        } else {
            print("ℹ️ MPS disabled")
        }
    }

    /// Load Metal shaders from the default library
    private func loadMetalShaders() -> Bool {
        guard let device = metalDevice, let library = library else { return false }
        
        do {
            // Load matrix multiply kernel
            matrixMultiplyFunc = library.makeFunction(name: "matrix_multiply")
            
            // Load softmax kernel
            softmaxFunc = library.makeFunction(name: "softmax")
            
            // Load add bias kernel
            addBiasFunc = library.makeFunction(name: "add_bias")
            
            // Load GELU activation kernel
            geluFunc = library.makeFunction(name: "gelu_activation")
            
            // Load embedding lookup kernel
            embeddingLookupFunc = library.makeFunction(name: "embedding_lookup")
            
            // Create compute pipelines
            if let matrixMultiplyFunc = matrixMultiplyFunc {
                matrixMultiplyPipeline = try device.makeComputePipelineState(function: matrixMultiplyFunc)
            }
            
            if let softmaxFunc = softmaxFunc {
                softmaxPipeline = try device.makeComputePipelineState(function: softmaxFunc)
            }
            
            if let addBiasFunc = addBiasFunc {
                addBiasPipeline = try device.makeComputePipelineState(function: addBiasFunc)
            }
            
            if let geluFunc = geluFunc {
                geluPipeline = try device.makeComputePipelineState(function: geluFunc)
            }
            
            if let embeddingLookupFunc = embeddingLookupFunc {
                embeddingLookupPipeline = try device.makeComputePipelineState(function: embeddingLookupFunc)
            }
            
            return true
        } catch {
            print("❌ Failed to load Metal shaders: \(error)")
            return false
        }
    }

    /// Metal-optimized version of predictNextToken
    func predictNextTokenMetal(inputIDs: [Int32], temperature: Float = 0.7) -> Int32? {
        // Check if Metal optimization is enabled and available
        guard useMetalOptimization,
              let device = metalDevice,
              let queue = commandQueue,
              let matrixMultiplyPipeline = matrixMultiplyPipeline,
              let softmaxPipeline = softmaxPipeline else {
            // Fall back to standard method if Metal not available/enabled
            return predictNextToken(inputIDs: inputIDs, temperature: temperature)
        }
        
        let paddedInput = padTokens(inputIDs, toLength: requiredSequenceLength)
        
        guard let inputArray = createMLMultiArray(paddedInput) else {
            return nil
        }
        
        do {
            // Use CoreML model with Metal acceleration
            let inputFeatures = try MLDictionaryFeatureProvider(dictionary: ["input_ids": inputArray])
            let output = try model.prediction(from: inputFeatures)
            
            // Get logits from output
            guard let outputKey = findOutputKey(in: output),
                  let outputValue = output.featureValue(for: outputKey),
                  let multiArray = outputValue.multiArrayValue else {
                return nil
            }
            
            // Extract the logits using Metal for softmax
            let tokenIndex = min(inputIDs.count - 1, requiredSequenceLength - 1)
            
            // Get shape information
            let shape = multiArray.shape.map { $0.intValue }
            
            // Ensure it's a 3D tensor [batch_size, seq_len, vocab_size]
            guard shape.count == 3, let vocabSize = shape.last else {
                return predictNextToken(inputIDs: inputIDs, temperature: temperature)
            }
            
            // Create a command buffer for all Metal operations
            guard let commandBuffer = queue.makeCommandBuffer() else {
                return predictNextToken(inputIDs: inputIDs, temperature: temperature)
            }
            
            // Extract logits for the token we want using Metal
            let logitsData = UnsafeMutablePointer<Float>(OpaquePointer(multiArray.dataPointer))
            let offset = tokenIndex * vocabSize
            
            // Create Metal buffers
            let logitsBuffer = device.makeBuffer(bytes: logitsData + offset,
                                                length: vocabSize * MemoryLayout<Float>.size,
                                                options: [])
            
            let outputBuffer = device.makeBuffer(length: vocabSize * MemoryLayout<Float>.size,
                                                options: [])
            
            // Set up softmax compute command
            guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
                return predictNextToken(inputIDs: inputIDs, temperature: temperature)
            }
            
            computeEncoder.setComputePipelineState(softmaxPipeline)
            computeEncoder.setBuffer(logitsBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(outputBuffer, offset: 0, index: 1)
            
            var vocabSizeUInt = UInt32(vocabSize)
            var stride = UInt32(1)
            computeEncoder.setBytes(&vocabSizeUInt, length: MemoryLayout<UInt32>.size, index: 2)
            computeEncoder.setBytes(&stride, length: MemoryLayout<UInt32>.size, index: 3)
            
            let threadsPerGrid = MTLSize(width: vocabSize, height: 1, depth: 1)
            let threadsPerThreadgroup = MTLSize(width: min(vocabSize, softmaxPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
            
            computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            computeEncoder.endEncoding()
            
            // Execute Metal commands
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            
            // Get the results
            guard let probabilities = outputBuffer?.contents() else {
                return predictNextToken(inputIDs: inputIDs, temperature: temperature)
            }
            
            let probabilitiesPointer = UnsafePointer<Float>(OpaquePointer(probabilities))
            let probabilitiesBuffer = UnsafeBufferPointer(start: probabilitiesPointer, count: vocabSize)
            let probsArray = Array(probabilitiesBuffer)
            
            // Sample from distribution or get argmax
            let predictedID: Int32
            if temperature > 0 && temperature != 1.0 {
                predictedID = Int32(sampleWithTemperatureMetal(from: probsArray, temperature: temperature))
            } else {
                predictedID = Int32(argmax(probsArray))
            }
            
            return predictedID
        } catch {
            print("❌ Metal inference error: \(error)")
            return predictNextToken(inputIDs: inputIDs, temperature: temperature)
        }
    }

    /// Sample with temperature using Metal
    private func sampleWithTemperatureMetal(from probabilities: [Float], temperature: Float) -> Int {
        // For Metal, we've already done the softmax, so just sample directly
        let rand = Float.random(in: 0..<1)
        var cumulative: Float = 0
        
        for (i, prob) in probabilities.enumerated() {
            cumulative += prob
            if rand < cumulative {
                return i
            }
        }
        
        // Fallback
        return probabilities.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
    }

    /// Find the output key in a MLFeatureProvider
    private func findOutputKey(in output: MLFeatureProvider) -> String? {
        let possibleKeys = ["logits", "var_3474", "var_3459"]
        for key in possibleKeys {
            if output.featureNames.contains(key) {
                return key
            }
        }
        return output.featureNames.first
    }
    /// Basic Metal matrix multiply implementation using custom shaders
    private func metalMatrixMultiply(A: MLMultiArray, B: MLMultiArray) -> MLMultiArray? {
        guard let device = metalDevice,
              let commandQueue = commandQueue,
              let pipeline = matrixMultiplyPipeline else {
            print("⚠️ Metal matrix multiply not available")
            return nil
        }
        
        // Get dimensions
        let M = A.shape[0].intValue  // Rows of A
        let K = A.shape[1].intValue  // Cols of A / Rows of B
        let N = B.shape[1].intValue  // Cols of B
        
        // Create result array
        guard let C = try? MLMultiArray(shape: [NSNumber(value: M), NSNumber(value: N)],
                                       dataType: .float32) else {
            print("❌ Failed to create result MLMultiArray")
            return nil
        }
        
        // Create Metal buffers
        guard let aBuffer = device.makeBuffer(bytes: A.dataPointer,
                                             length: M * K * MemoryLayout<Float>.size,
                                             options: []),
              let bBuffer = device.makeBuffer(bytes: B.dataPointer,
                                             length: K * N * MemoryLayout<Float>.size,
                                             options: []),
              let cBuffer = device.makeBuffer(length: M * N * MemoryLayout<Float>.size,
                                             options: []) else {
            print("❌ Failed to create Metal buffers")
            return nil
        }
        
        // Create command buffer and encoder
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            print("❌ Failed to create command buffer or encoder")
            return nil
        }
        
        // Set up compute command
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(aBuffer, offset: 0, index: 0)
        encoder.setBuffer(bBuffer, offset: 0, index: 1)
        encoder.setBuffer(cBuffer, offset: 0, index: 2)
        
        // Set dimensions as constants (matching your Metal shader)
        var mDim = UInt32(M)
        var nDim = UInt32(N)
        var kDim = UInt32(K)
        encoder.setBytes(&mDim, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&nDim, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&kDim, length: MemoryLayout<UInt32>.size, index: 5)
        
        // Dispatch threads - 2D grid matching your Metal kernel
        let threadsPerGrid = MTLSize(width: N, height: M, depth: 1)
        let maxThreadsPerGroup = pipeline.maxTotalThreadsPerThreadgroup
        let threadsPerThreadgroup = MTLSize(
            width: min(16, N, Int(sqrt(Double(maxThreadsPerGroup)))),
            height: min(16, M, Int(sqrt(Double(maxThreadsPerGroup)))),
            depth: 1
        )
        
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        // Execute and wait
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Check for errors
        if commandBuffer.status == .error {
            print("❌ Metal command buffer failed")
            return nil
        }
        
        // Copy result back to MLMultiArray
        let cPointer = UnsafeMutablePointer<Float>(OpaquePointer(C.dataPointer))
        let resultPointer = UnsafeMutablePointer<Float>(OpaquePointer(cBuffer.contents()))
        
        for i in 0..<(M * N) {
            cPointer[i] = resultPointer[i]
        }
        
        print("✅ Metal matrix multiply completed: (\(M)x\(K)) * (\(K)x\(N)) = (\(M)x\(N))")
        return C
    }
    private func mpsMatrixMultiply(A: MLMultiArray, B: MLMultiArray) -> MLMultiArray? {
        guard useMPS,
              let device = metalDevice,
              let commandQueue = commandQueue,
              device.supportsFamily(.apple3) else {
            // Fall back to custom Metal implementation
            return metalMatrixMultiply(A: A, B: B)
        }
        
        // Get dimensions
        let M = A.shape[0].intValue
        let K = A.shape[1].intValue
        let N = B.shape[1].intValue
        
        // Create result array
        guard let C = try? MLMultiArray(shape: [NSNumber(value: M), NSNumber(value: N)],
                                       dataType: .float32) else {
            return nil
        }
        
        // Create MPS matrix multiply kernel
        let matrixMultiply = MPSMatrixMultiplication(device: device,
                                                    transposeLeft: false,
                                                    transposeRight: false,
                                                    resultRows: M,
                                                    resultColumns: N,
                                                    interiorColumns: K,
                                                    alpha: 1.0,
                                                    beta: 0.0)
        
        // Create MPS matrices
        let aDesc = MPSMatrixDescriptor(rows: M, columns: K, rowBytes: K * MemoryLayout<Float>.size,
                                       dataType: .float32)
        let bDesc = MPSMatrixDescriptor(rows: K, columns: N, rowBytes: N * MemoryLayout<Float>.size,
                                       dataType: .float32)
        let cDesc = MPSMatrixDescriptor(rows: M, columns: N, rowBytes: N * MemoryLayout<Float>.size,
                                       dataType: .float32)
        
        let aMatrix = MPSMatrix(buffer: device.makeBuffer(bytes: A.dataPointer,
                                                         length: M * K * MemoryLayout<Float>.size,
                                                         options: [])!,
                               descriptor: aDesc)
        
        let bMatrix = MPSMatrix(buffer: device.makeBuffer(bytes: B.dataPointer,
                                                         length: K * N * MemoryLayout<Float>.size,
                                                         options: [])!,
                               descriptor: bDesc)
        
        let cBuffer = device.makeBuffer(length: M * N * MemoryLayout<Float>.size, options: [])!
        let cMatrix = MPSMatrix(buffer: cBuffer, descriptor: cDesc)
        
        // Create command buffer and encode operation
        let commandBuffer = commandQueue.makeCommandBuffer()!
        matrixMultiply.encode(commandBuffer: commandBuffer, leftMatrix: aMatrix, rightMatrix: bMatrix,
                             resultMatrix: cMatrix)
        
        // Commit and wait
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Copy result back to MLMultiArray
        let cPointer = UnsafeMutablePointer<Float>(OpaquePointer(C.dataPointer))
        let resultPointer = UnsafeMutablePointer<Float>(OpaquePointer(cBuffer.contents()))
        
        for i in 0..<(M * N) {
            cPointer[i] = resultPointer[i]
        }
        
        return C
    }
    
    /// Update generateStreaming to use Metal acceleration when enabled
    func generateStreaming(
        inputIDs: [Int32],
        maxTokens: Int = 20,
        eosTokenID: Int32? = nil,
        temperature: Float = 0.2
    ) -> [Int32] {
        resetCache()
        
        // Ensure input isn't empty and fits within required length
        var input: [Int32] = []
        if inputIDs.isEmpty {
            input = [0] // Use at least one token (padding token)
        } else if inputIDs.count > requiredSequenceLength {
            input = Array(inputIDs.suffix(requiredSequenceLength))
        } else {
            input = inputIDs
        }
        
        var generated: [Int32] = []
        
        // Generate tokens one by one
        for _ in 0..<maxTokens {
            let nextToken: Int32?
            
            if useMetalOptimization {
                nextToken = predictNextTokenMetal(inputIDs: input, temperature: temperature)
            } else {
                nextToken = predictNextToken(inputIDs: input, temperature: temperature)
            }
            
            guard let token = nextToken else {
                print("⚠️ Failed to generate token")
                break
            }
            
            generated.append(token)
            print("▶️ Generated token: \(token)")
            
            // For next iteration:
            // 1. Add new token to input
            input = input + [token]
            
            // 2. If input gets too long, remove oldest tokens to keep length <= requiredSequenceLength
            if input.count > requiredSequenceLength {
                input = Array(input.suffix(requiredSequenceLength))
            }
            
            // 3. Check for EOS token
            if let eos = eosTokenID, token == eos {
                print("🛑 Reached EOS token")
                break
            }
        }
        
        return generated
    }
    // MARK: - Token Management
    
    func padTokens(_ tokens: [Int32], toLength targetLength: Int, padID: Int32 = 0) -> [Int32] {
        if tokens.count >= targetLength {
            // Strictly limit to the target length
            return Array(tokens.prefix(targetLength))
        } else {
            // Pad to the exact target length
            return tokens + Array(repeating: padID, count: targetLength - tokens.count)
        }
    }
    
    // MARK: - MLMultiArray Utilities
    
    func createMLMultiArray(_ input: [Int32]) -> MLMultiArray? {
        // Always create with the exact required shape
        guard let mlArray = try? MLMultiArray(shape: [1, NSNumber(value: requiredSequenceLength)], dataType: .int32) else {
            return nil
        }
        
        // Make sure input has exactly requiredSequenceLength tokens
        let adjustedInput = padTokens(input, toLength: requiredSequenceLength)
        
        for (index, value) in adjustedInput.enumerated() {
            mlArray[index] = NSNumber(value: value)
        }
        return mlArray
    }
    
    private func argmax(_ array: [Float]) -> Int {
        var maxValue = array[0]
        var maxIndex = 0
        for i in 1..<array.count {
            if array[i] > maxValue {
                maxValue = array[i]
                maxIndex = i
            }
        }
        return maxIndex
    }
    
    func sampleWithTemperature(from logits: [Float], temperature: Float = 1.0) -> Int {
        // Apply temperature scaling
        var scaledLogits = logits
        if temperature > 0 && temperature != 1.0 {
            scaledLogits = logits.map { $0 / temperature }
        }
        
        // Convert to probabilities with softmax
        let maxLogit = scaledLogits.max() ?? 0
        let expLogits = scaledLogits.map { exp($0 - maxLogit) }
        let sum = expLogits.reduce(0, +)
        let probs = expLogits.map { $0 / sum }
        
        // Sample from distribution
        let rand = Float.random(in: 0..<1)
        var cumulative: Float = 0
        for (i, prob) in probs.enumerated() {
            cumulative += prob
            if rand < cumulative {
                return i
            }
        }
        return probs.count - 1
    }

    /// Extracts the logits for a specific token index (usually the last non-padded token)
    /// from a 3D MLMultiArray of shape [1, sequenceLength, vocabSize].
    /// Returns a [Float] of length == vocabSize for the selected token.
    func extractLastTokenLogits(from multiArray: MLMultiArray, atToken tokenIndex: Int) -> [Float]? {
        // Step 1: Get shape
        let shape = multiArray.shape.map { $0.intValue }
        guard shape.count == 3 else {
            print("❌ Expected 3D MLMultiArray, got shape: \(shape)")
            return nil
        }

        let batchSize = shape[0]         // usually 1
        let sequenceLength = shape[1]    // e.g. 10
        let vocabSize = shape[2]         // e.g. 32000

        // Step 2: Validate token index
        guard tokenIndex < sequenceLength else {
            print("❌ tokenIndex \(tokenIndex) out of range (max: \(sequenceLength - 1))")
            return nil
        }

        // Step 3: Calculate flat offset
        let offset = tokenIndex * vocabSize

        // Step 4: Get pointer to underlying Float32 data
        let pointer = UnsafeMutablePointer<Float>(OpaquePointer(multiArray.dataPointer))

        // Step 5: Extract logits for selected token
        let buffer = UnsafeBufferPointer(start: pointer + offset, count: vocabSize)
        return Array(buffer)
    }
    
    // MARK: - Cache Management
    
    func resetCache() {
        kvCache = nil
        print("🔄 KV Cache reset")
    }
    
    func updateCache(with newCacheValues: [String: MLMultiArray]?) {
        guard let newCache = newCacheValues, !newCache.isEmpty else {
            return
        }
        
        kvCache = newCache  // Simply replace cache for now
        //print("📦 Updated KV Cache with \(newCache.count) keys")
    }
    
    // MARK: - Predict next token ID only
    func predictNextToken(inputIDs: [Int32], temperature: Float = 0.7) -> Int32? {
        // Always enforce the exact required sequence length
        let paddedInput = padTokens(inputIDs, toLength: requiredSequenceLength)
        
        //print("🧠 Model input: \(paddedInput) (length: \(paddedInput.count))")

        guard let inputArray = createMLMultiArray(paddedInput) else {
            print("❌ Failed to create MLMultiArray")
            return nil
        }

        do {
            // Build input features
            let inputFeatures = try MLDictionaryFeatureProvider(dictionary: ["input_ids": inputArray])
            let output = try model.prediction(from: inputFeatures)

            // Get cache if available (though your model might not support this)
            var newCache: [String: MLMultiArray] = [:]
            for featureName in output.featureNames {
                if featureName.contains("key_cache") || featureName.contains("value_cache") {
                    if let cacheValue = output.featureValue(for: featureName)?.multiArrayValue {
                        newCache[featureName] = cacheValue
                    }
                }
            }
            updateCache(with: newCache)

            // Get logits from output - try common output names or use the first key
            let possibleLogitsKeys = ["logits", "output_logits", "var_3459"]
            var outputKey: String? = nil
            
            for key in possibleLogitsKeys {
                if output.featureNames.contains(where: { $0.contains(key) || $0 == key }) {
                    outputKey = output.featureNames.first(where: { $0.contains(key) || $0 == key })
                    break
                }
            }
            
            if outputKey == nil {
                outputKey = output.featureNames.first
                //print("⚠️ Using fallback output key: \(outputKey ?? "none found")")
            }
            
            guard let outputKey = outputKey,
                  let outputValue = output.featureValue(for: outputKey),
                  let multiArray = outputValue.multiArrayValue else {
                print("❌ Failed to retrieve logits from model output. Available keys: \(output.featureNames.joined(separator: ", "))")
                return nil
            }

            // Use the index of the last real token (not padding)
            let realTokenCount = min(inputIDs.count, requiredSequenceLength)
            let tokenIndex = realTokenCount - 1
            
            //print("📍 Using logits from token index: \(tokenIndex)")
            
            guard let logits = extractLastTokenLogits(from: multiArray, atToken: tokenIndex) else {
                print("❌ Failed to extract logits at token index \(tokenIndex)")
                return nil
            }

            // Use temperature sampling or argmax
            let predictedID: Int32
            if temperature > 0 && temperature != 1.0 {
                predictedID = Int32(sampleWithTemperature(from: logits, temperature: temperature))
            } else {
                predictedID = Int32(argmax(logits))
            }
            
            //print("🔮 Predicted token ID: \(predictedID)")
            return predictedID

        } catch {
            print("❌ Inference error: \(error)")
            return nil
        }
    }

    // MARK: - Streaming generation
    /*
     func generateStreaming(
        inputIDs: [Int32],
        maxTokens: Int = 20,
        eosTokenID: Int32? = nil,
        temperature: Float = 0.2
    ) -> [Int32] {
        resetCache()
        
        // Ensure input isn't empty and fits within required length
        var input: [Int32] = []
        if inputIDs.isEmpty {
            print("⚠️ Empty input IDs provided, using padding")
            input = [0] // Use at least one token (padding token)
        } else if inputIDs.count > requiredSequenceLength {
            print("⚠️ Input too long (\(inputIDs.count)), truncating to \(requiredSequenceLength)")
            input = Array(inputIDs.suffix(requiredSequenceLength))
        } else {
            input = inputIDs
        }
        
        var generated: [Int32] = []
        
        print("🚀 Starting generation with \(input.count) input tokens, temperature: \(temperature)")
        if let eos = eosTokenID {
            print("⏹️ Will stop at EOS token ID: \(eos)")
        }

        // Generate tokens one by one
        for _ in 0..<maxTokens {
            guard let nextToken = predictNextToken(inputIDs: input, temperature: temperature) else {
                print("⚠️ Failed to generate token")
                break
            }
            
            generated.append(nextToken)
            print("▶️ Generated token: \(nextToken)")
            
            // For next iteration:
            // 1. Add new token to input
            input = input + [nextToken]
            
            // 2. If input gets too long, remove oldest tokens to keep length <= requiredSequenceLength
            if input.count > requiredSequenceLength {
                input = Array(input.suffix(requiredSequenceLength))
            }
            
            // 3. Check for EOS token
            if let eos = eosTokenID, nextToken == eos {
                print("🛑 Reached EOS token")
                break
            }
        }

        return generated
    }
    */
    // MARK: - Legacy method for backward compatibility
    func predict(inputIDs: [Int32]) -> String {
        // Ensure we use exactly requiredSequenceLength tokens
        let paddedInput = padTokens(inputIDs, toLength: requiredSequenceLength)
        let tokenIndex = min(inputIDs.count, requiredSequenceLength) - 1

        guard let inputArray = createMLMultiArray(paddedInput) else {
            return "❌ Failed to create MLMultiArray"
        }

        do {
            let inputFeatures = try MLDictionaryFeatureProvider(dictionary: ["input_ids": inputArray])
            let output = try model.prediction(from: inputFeatures)

            // Print all output keys
            //print("✅ Output keys:", output.featureNames)

            // Automatically use the first available key (likely var_3459)
            guard let outputKey = output.featureNames.first else {
                return "❌ No output keys found"
            }

            guard let outputValue = output.featureValue(for: outputKey) else {
                return "❌ Feature value for key '\(outputKey)' is nil"
            }

            guard let multiArray = outputValue.multiArrayValue else {
                return "❌ Feature '\(outputKey)' is not a MultiArray"
            }
            
            // print("🧠 Model input (padded to \(requiredSequenceLength)): \(paddedInput)")
            // print("📍 Extracting logits at token index: \(tokenIndex)")
            
            // Extract final token logits
            guard let logits = extractLastTokenLogits(from: multiArray, atToken: tokenIndex) else {
                return "❌ Failed to extract last-token logits"
            }

            let maxIndex = argmax(logits)
            return "🔮 Predicted token ID: \(maxIndex)"

        } catch {
            return "❌ Inference error: \(error)"
        }
    }
}
