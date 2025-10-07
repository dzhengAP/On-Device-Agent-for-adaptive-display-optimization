import Foundation
import CoreML
import Metal
import QuartzCore

/// Simple inference performance tester for different acceleration methods
class InferenceBenchmark {
    
    enum AccelerationMethod: String, CaseIterable {
        case cpuOnly = "CPU Only"
        case coreMLDefault = "CoreML Default"
        case metalOptimized = "Metal Optimized"
        case mps = "Metal Performance Shaders"
    }
    
    struct InferenceResult {
        let method: AccelerationMethod
        let inferenceTimeMs: Double
        let tokensGenerated: Int
        let tokensPerSecond: Double
        let memoryUsageMB: Double
        let success: Bool
    }
    
    private let metalDevice: MTLDevice?
    private let deviceInfo: String
    
    init() {
        metalDevice = MTLCreateSystemDefaultDevice()
        if let device = metalDevice {
            deviceInfo = device.name
        } else {
            deviceInfo = "No Metal device available"
        }
    }
    
    /// Run a quick inference benchmark across all acceleration methods
    func runInferenceBenchmark() {
        let separator = String(repeating: "=", count: 60)
        print("\n\(separator)")
        print("🚀 INFERENCE PERFORMANCE BENCHMARK")
        print("Device: \(deviceInfo)")
        print(separator)
        
        var results: [InferenceResult] = []
        
        // Test each acceleration method
        for method in AccelerationMethod.allCases {
            print("\n⚡ Testing \(method.rawValue)...")
            
            let result = testSingleInference(method: method)
            results.append(result)
            
            if result.success {
                print("✅ Success: \(String(format: "%.1f", result.inferenceTimeMs))ms, \(result.tokensGenerated) tokens, \(String(format: "%.1f", result.tokensPerSecond)) tok/s")
            } else {
                print("❌ Failed")
            }
            
            // Small delay between tests to avoid thermal issues
            usleep(1000000) // 1 second in microseconds
        }
        
        // Print summary
        printBenchmarkSummary(results)
    }
    
    /// Test a single inference with specified acceleration method
    private func testSingleInference(method: AccelerationMethod) -> InferenceResult {
        let startTime = CACurrentMediaTime()
        let memoryBefore = getMemoryUsage()
        
        // Create engine with specific configuration
        guard let engine = createConfiguredEngine(for: method) else {
            return InferenceResult(
                method: method,
                inferenceTimeMs: 0,
                tokensGenerated: 0,
                tokensPerSecond: 0,
                memoryUsageMB: 0,
                success: false
            )
        }
        
        // Load vocab for tokenization
        guard let vocab = loadVocab() else {
            return InferenceResult(
                method: method,
                inferenceTimeMs: 0,
                tokensGenerated: 0,
                tokensPerSecond: 0,
                memoryUsageMB: 0,
                success: false
            )
        }
        
        // Test prompt
        let testPrompt = "gaming app brightness:"
        let tokens = tokenizeText(testPrompt, vocab: vocab)
        
        print("  🔤 Input: \"\(testPrompt)\" -> \(tokens.count) tokens")
        
        // Generate tokens
        let generatedIDs = engine.generateStreaming(
            inputIDs: tokens,
            maxTokens: 15,
            temperature: 0.1
        )
        
        let endTime = CACurrentMediaTime()
        let memoryAfter = getMemoryUsage()
        
        let totalTime = (endTime - startTime) * 1000 // Convert to ms
        let memoryDelta = max(0, memoryAfter - memoryBefore)
        let tokensPerSecond = generatedIDs.count > 0 ? Double(generatedIDs.count) / ((endTime - startTime)) : 0
        
        // Convert generated tokens back to text for verification
        let generatedText = convertTokensToText(generatedIDs, vocab: vocab)
        print("  📝 Generated: \"\(generatedText)\"")
        
        return InferenceResult(
            method: method,
            inferenceTimeMs: totalTime,
            tokensGenerated: generatedIDs.count,
            tokensPerSecond: tokensPerSecond,
            memoryUsageMB: memoryDelta,
            success: generatedIDs.count > 0
        )
    }
    
    /// Create an LLMEngine configured for the specified acceleration method
    private func createConfiguredEngine(for method: AccelerationMethod) -> LLMEngine? {
        guard let engine = LLMEngine() else {
            print("    ❌ Failed to create LLMEngine")
            return nil
        }
        
        // Configure the engine based on the method
        switch method {
        case .cpuOnly:
            engine.setComputeUnits(.cpuOnly)
            engine.setMetalOptimizationEnabled(false)
            engine.setMPSEnabled(false)
            
        case .coreMLDefault:
            engine.setComputeUnits(.all)
            engine.setMetalOptimizationEnabled(false)
            engine.setMPSEnabled(false)
            
        case .metalOptimized:
            engine.setComputeUnits(.cpuAndGPU)
            engine.setMetalOptimizationEnabled(true)
            engine.setMPSEnabled(false)
            
        case .mps:
            engine.setComputeUnits(.cpuAndGPU)
            engine.setMetalOptimizationEnabled(true)
            engine.setMPSEnabled(true)
        }
        
        print("    ⚙️ Configured for \(method.rawValue)")
        return engine
    }
    
    /// Print benchmark summary
    private func printBenchmarkSummary(_ results: [InferenceResult]) {
        let separator = String(repeating: "=", count: 60)
        let dashLine = String(repeating: "-", count: 60)
        
        print("\n\(separator)")
        print("📊 BENCHMARK SUMMARY")
        print(separator)
        
        let successfulResults = results.filter { $0.success }
        
        if successfulResults.isEmpty {
            print("❌ No successful inference tests")
            return
        }
        
        // Sort by inference time (fastest first)
        let sortedResults = successfulResults.sorted { $0.inferenceTimeMs < $1.inferenceTimeMs }
        
        print(String(format: "%-20s %10s %8s %10s %8s", "Method", "Time(ms)", "Tokens", "Tok/s", "Memory(MB)"))
        print(dashLine)
        
        for result in sortedResults {
            print(String(format: "%-20s %10.1f %8d %10.1f %8.1f",
                         result.method.rawValue,
                         result.inferenceTimeMs,
                         result.tokensGenerated,
                         result.tokensPerSecond,
                         result.memoryUsageMB))
        }
        
        // Find best performers
        if let fastest = sortedResults.first {
            print("\n🏆 Fastest: \(fastest.method.rawValue) (\(String(format: "%.1f", fastest.inferenceTimeMs))ms)")
        }
        
        if let bestThroughput = sortedResults.max(by: { $0.tokensPerSecond < $1.tokensPerSecond }) {
            print("⚡ Highest throughput: \(bestThroughput.method.rawValue) (\(String(format: "%.1f", bestThroughput.tokensPerSecond)) tok/s)")
        }
        
        if let lowestMemory = sortedResults.min(by: { $0.memoryUsageMB < $1.memoryUsageMB }) {
            print("💾 Lowest memory: \(lowestMemory.method.rawValue) (\(String(format: "%.1f", lowestMemory.memoryUsageMB))MB)")
        }
        
        print(separator)
    }
    
    // MARK: - Utility Functions
    
    private func getMemoryUsage() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let kerr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        if kerr == KERN_SUCCESS {
            return Double(info.resident_size) / (1024 * 1024)
        }
        return 0.0
    }
    
    private func loadVocab() -> [String: Int32]? {
        guard let path = Bundle.main.path(forResource: "vocab", ofType: "json"),
              let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
            return nil
        }
        do {
            let decoder = JSONDecoder()
            return try decoder.decode([String: Int32].self, from: data)
        } catch {
            return nil
        }
    }
    
    private func tokenizeText(_ text: String, vocab: [String: Int32]) -> [Int32] {
        let hasSentencePiece = vocab.keys.contains { $0.hasPrefix("▁") }
        let unkID = vocab["<unk>"] ?? vocab["UNK"] ?? 0
        
        if hasSentencePiece {
            var tokens: [String] = []
            let words = text.split(separator: " ")
            for (i, word) in words.enumerated() {
                let token = i == 0 ? String(word) : "▁" + String(word)
                tokens.append(token)
            }
            return tokens.map { vocab[$0] ?? vocab["▁" + $0] ?? unkID }
        } else {
            let tokens = text.split(separator: " ").map { String($0) }
            return tokens.map { vocab[$0] ?? unkID }
        }
    }
    
    private func convertTokensToText(_ tokens: [Int32], vocab: [String: Int32]) -> String {
        let idToToken = Dictionary(uniqueKeysWithValues: vocab.map { ($0.value, $0.key) })
        let hasSentencePiece = vocab.keys.contains { $0.hasPrefix("▁") }
        
        var result = ""
        for id in tokens {
            guard let token = idToToken[id] else { continue }
            
            if ["<s>", "</s>", "<|endoftext|>", "<unk>", "UNK"].contains(token) {
                continue
            } else if hasSentencePiece && token.hasPrefix("▁") {
                result += " " + String(token.dropFirst())
            } else {
                result += token
            }
        }
        
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// MARK: - Integration Helper

/// Call this function to run the benchmark from anywhere in your app
func runQuickInferenceBenchmark() {
    let benchmark = InferenceBenchmark()
    benchmark.runInferenceBenchmark()
}
