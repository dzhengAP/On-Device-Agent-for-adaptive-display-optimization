import Foundation

/// Result tracking for experimental analysis
struct EngineResult {
    let timestamp: Date
    let appName: String
    let appType: String
    let settingName: String
    let predictedValue: String
    let confidence: Double
    let executionTimeMs: Double
    let memoryUsageMB: Double
    let source: EngineSource
    
    enum EngineSource {
        case llm
        case rag
        case hybrid
    }
}

/// Result collector for experimental analysis
class ResultCollector {
    static let shared = ResultCollector()
    private var results: [EngineResult] = []
    private let queue = DispatchQueue(label: "result.collector", qos: .utility)
    
    private init() {}
    
    func addResult(_ result: EngineResult) {
        queue.async {
            self.results.append(result)
        }
    }
    
    func getResults(completion: @escaping ([EngineResult]) -> Void) {
        queue.async {
            completion(self.results)
        }
    }
    
    func clearResults() {
        queue.async {
            self.results.removeAll()
        }
    }
}

/// Hybrid engine that combines RAG knowledge and LLM fine-tuning
class HybridSettingsEngine {
    /// The underlying LLM engine
    private let llmEngine: LLMEngine
    
    /// Database for RAG capabilities
    private let database = DisplaySettingsDatabase.shared
    
    /// Result tracking
    private let resultCollector = ResultCollector.shared
    
    /// For debugging and experimental analysis
    private var lastPrompt: String = ""
    private var lastOutput: String = ""
    private var lastContext: String = ""
    private var lastExecutionTime: Double = 0.0
    private var lastMemoryUsage: Double = 0.0
    
    /// Performance metrics
    private var totalRequests: Int = 0
    private var successfulPredictions: Int = 0
    private var totalExecutionTime: Double = 0.0
    
    /// Initializes the hybrid engine
    init?() {
        // Try to create the LLM engine
        guard let engine = LLMEngine() else { return nil }
        self.llmEngine = engine
        print("🚀 HybridSettingsEngine initialized with LLMEngine")
    }
    
    /// Gets recommendations using the hybrid approach
    /// - Parameters:
    ///   - appName: Name of the application
    ///   - appType: Type/category of the application
    /// - Returns: String containing recommended settings
    func getRecommendations(for appName: String, appType: String) -> String {
        let startTime = CFAbsoluteTimeGetCurrent()
        let memoryBefore = getMemoryUsage()
        
        print("⚙️ Getting hybrid recommendations for \(appName) (\(appType))")
        
        // Try single LLM call first with timeout
        if let llmResult = getSingleLLMRecommendation(for: appType) {
            let executionTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
            updatePerformanceMetrics(executionTime: executionTime, success: true)
            return llmResult
        }
        
        // Fall back to RAG system
        print("⚠️ LLM failed, using RAG fallback")
        let ragResult = database.getFormattedSettings(for: appName, appType: appType)
        
        let executionTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        updatePerformanceMetrics(executionTime: executionTime, success: !ragResult.isEmpty)
        
        return ragResult
    }
    
    /// Single LLM call for all settings with timeout
    private func getSingleLLMRecommendation(for appType: String) -> String? {
        let prompt = """
    \(appType) settings:
    brightness: 75%
    contrast: 80%
    color_temperature: 6500K
    blue_light_filter: 30%
    text_sharpness: 85%
    refresh_rate: 60Hz
    response_time: 5ms
    hdr: enabled
    """
        
        let startTime = CFAbsoluteTimeGetCurrent()
        let tokens = tokenizePrompt(prompt)
        
        let generatedIDs = llmEngine.generateStreaming(
            inputIDs: tokens,
            maxTokens: 30,  // Reduced from 15
            temperature: 0.3  // Increased from 0.1
        )
        
        // Check for timeout (2 seconds max)
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        if elapsed > 2.0 {
            print("⏰ LLM timeout after \(elapsed) seconds")
            return nil
        }
        
        if generatedIDs.isEmpty {
            print("⚠️ LLM generated no tokens")
            return nil
        }
        
        let output = convertTokensToString(generatedIDs)
        return parseStructuredOutput(output, appType: appType)
    }
    
    /// Parse structured LLM output into formatted settings
    private func parseStructuredOutput(_ output: String, appType: String) -> String? {
        var result = ""
        let lines = output.components(separatedBy: .newlines)
        
        for line in lines {
            if let colonIndex = line.firstIndex(of: ":") {
                let key = String(line[..<colonIndex]).trimmingCharacters(in: .whitespacesAndNewlines)
                let value = String(line[line.index(after: colonIndex)...]).trimmingCharacters(in: .whitespacesAndNewlines)
                
                if !key.isEmpty && !value.isEmpty && !value.contains("should") && !value.contains("app") {
                    let formattedKey = key.replacingOccurrences(of: "_", with: " ").capitalized
                    result += "\(formattedKey): \(value)\n"
                }
            }
        }
        
        // Only return if we got at least 3 valid settings
        let settingsCount = result.components(separatedBy: "\n").filter { !$0.isEmpty }.count
        return settingsCount >= 3 ? result.trimmingCharacters(in: .whitespacesAndNewlines) : nil
    }
    
    /// Calculate confidence score for LLM predictions
    private func calculateLLMConfidence(_ value: String, settingType: String) -> Double {
        // Base confidence on value validity and format
        var confidence = 0.5
        
        // Check if value matches expected format for setting type
        if isValidSettingFormat(value, for: settingType) {
            confidence += 0.3
        }
        
        // Check if value is within reasonable range
        if isReasonableSettingValue(value, for: settingType) {
            confidence += 0.2
        }
        
        // Bonus for specific formats that indicate higher confidence
        if value.contains("%") || value.contains("K") || value.contains("Hz") || value.contains("ms") {
            confidence += 0.1
        }
        
        return min(1.0, confidence)
    }
    
    /// Validate setting format
    private func isValidSettingFormat(_ value: String, for settingType: String) -> Bool {
        let lowercaseValue = value.lowercased()
        let lowercaseType = settingType.lowercased()
        
        switch lowercaseType {
        case let type where type.contains("brightness") || type.contains("contrast"):
            return lowercaseValue.contains("%")
        case let type where type.contains("temperature"):
            return lowercaseValue.contains("k")
        case let type where type.contains("refresh"):
            return lowercaseValue.contains("hz")
        case let type where type.contains("response"):
            return lowercaseValue.contains("ms")
        case let type where type.contains("hdr"):
            return lowercaseValue.contains("enable") || lowercaseValue.contains("disable") || lowercaseValue.contains("auto")
        default:
            return !lowercaseValue.contains("error") && !lowercaseValue.isEmpty
        }
    }
    
    /// Check if setting value is reasonable
    private func isReasonableSettingValue(_ value: String, for settingType: String) -> Bool {
        guard let numValue = extractNumericValue(from: value) else {
            // For non-numeric values, check common patterns
            let lowercaseValue = value.lowercased()
            return !lowercaseValue.contains("error") && !lowercaseValue.isEmpty
        }
        
        // Check numeric ranges for different setting types
        let lowercaseType = settingType.lowercased()
        
        switch lowercaseType {
        case let type where type.contains("brightness") || type.contains("contrast"):
            return numValue >= 0 && numValue <= 100
        case let type where type.contains("temperature"):
            return numValue >= 2000 && numValue <= 10000
        case let type where type.contains("refresh"):
            return numValue >= 24 && numValue <= 360
        case let type where type.contains("response"):
            return numValue >= 1 && numValue <= 50
        case let type where type.contains("gamma"):
            return numValue >= 1.0 && numValue <= 3.0
        default:
            return numValue > 0
        }
    }
    
    /// Update performance tracking metrics
    private func updatePerformanceMetrics(executionTime: Double, success: Bool) {
        totalRequests += 1
        totalExecutionTime += executionTime
        
        if success {
            successfulPredictions += 1
        }
        
        // Store for debug access
        lastExecutionTime = executionTime
        lastMemoryUsage = getMemoryUsage()
    }
    
    /// Get performance statistics for experimental analysis
    func getPerformanceStats() -> (requests: Int, successRate: Double, avgExecutionTime: Double) {
        let successRate = totalRequests > 0 ? Double(successfulPredictions) / Double(totalRequests) : 0.0
        let avgTime = totalRequests > 0 ? totalExecutionTime / Double(totalRequests) : 0.0
        
        return (totalRequests, successRate, avgTime)
    }
    
    /// Reset performance tracking
    func resetPerformanceStats() {
        totalRequests = 0
        successfulPredictions = 0
        totalExecutionTime = 0.0
    }
    
    /// Attempt to get a setting value using the fine-tuned LLM
    /// - Parameters:
    ///   - setting: The setting to get (e.g. "brightness")
    ///   - appType: Type of application
    ///   - app: Name of the application
    /// - Returns: Formatted setting value if successful, nil otherwise
    private func getSettingFromLLM(_ setting: String, for appType: String, app: String) -> String? {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Create more specific, completion-style prompts
        let prompt: String
        
        switch setting {
        case "brightness":
            prompt = "\(appType) brightness: "
        case "contrast":
            prompt = "\(appType) contrast: "
        case "color temperature":
            prompt = "\(appType) color temperature: "
        case "refresh rate":
            prompt = "\(appType) refresh rate: "
        case "blue light filter":
            prompt = "\(appType) blue light filter: "
        case "gamma":
            prompt = "\(appType) gamma: "
        case "response time":
            prompt = "\(appType) response time: "
        case "text sharpness":
            prompt = "\(appType) text sharpness: "
        case "HDR":
            prompt = "\(appType) HDR: "
        default:
            prompt = "\(appType) \(setting): "
        }
        
        self.lastPrompt = prompt
        
        // Get RAG context for better prompting
        let context = database.getRAGContext(for: app, appType: appType)
        self.lastContext = context
        
        // Use just the simple prompt without RAG context for better completion
        let tokens = tokenizePrompt(prompt)
        let generatedIDs = llmEngine.generateStreaming(
            inputIDs: tokens,
            maxTokens: 15,  // Increased from 5
            temperature: 0.1  // Slightly higher for more variety
        )
        
        // Check for empty output
        if generatedIDs.isEmpty {
            print("⚠️ LLM generated no tokens for \(setting)")
            return nil
        }
        
        // Convert tokens back to text
        let output = convertTokensToString(generatedIDs)
        self.lastOutput = output
        print("📝 Raw LLM output for \(setting): \(output)")
        
        // Try to interpret the output in the context of the setting
        if let interpretedValue = interpretSettingValue(output, settingType: setting) {
            print("✅ Interpreted \(setting) as: \(interpretedValue)")
            return interpretedValue
        }
        
        // Enhanced number extraction
        if let extractedValue = extractNumericValue(from: output, settingType: setting) {
            print("✅ Extracted \(setting) as: \(extractedValue)")
            return extractedValue
        }
        
        print("⚠️ Could not extract a value for \(setting) from LLM output")
        return nil
    }
    
    /// Enhanced numeric value extraction
    private func extractNumericValue(from text: String, settingType: String) -> String? {
        // Multiple extraction patterns
        let patterns = [
            "^(\\d+)",                    // Number at start
            "\\b(\\d+)\\b",              // Standalone number
            "(\\d+)%",                   // Percentage
            "(\\d+)K",                   // Kelvin
            "(\\d+)Hz",                  // Hertz
            "(\\d+)ms",                  // Milliseconds
            "(\\d+\\.\\d+)",             // Decimal numbers
            "\\s+(\\d+)\\s*$",           // Number at end with spaces
            ":\\s*(\\d+)",               // Number after colon
        ]
        
        for pattern in patterns {
            if let regex = try? NSRegularExpression(pattern: pattern),
               let match = regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)),
               let numberRange = Range(match.range(at: 1), in: text) {
                let numberStr = String(text[numberRange])
                if let number = Int(numberStr), isValidValueForSetting(number, settingType: settingType) {
                    return formatValue(numberStr, for: settingType)
                }
            }
        }
        
        return nil
    }
    
    /// Extract numeric value for validation
    private func extractNumericValue(from text: String) -> Double? {
        let pattern = try? NSRegularExpression(pattern: "\\d+\\.?\\d*")
        let range = NSRange(text.startIndex..., in: text)
        
        if let match = pattern?.firstMatch(in: text, range: range),
           let numberRange = Range(match.range, in: text) {
            return Double(String(text[numberRange]))
        }
        return nil
    }
    
    /// Check if a numeric value is valid for a given setting type
    private func isValidValueForSetting(_ value: Int, settingType: String) -> Bool {
        switch settingType.lowercased() {
        case "brightness", "contrast", "blue light filter", "text sharpness":
            return value >= 0 && value <= 100
        case "color temperature":
            return value >= 2000 && value <= 10000
        case "refresh rate":
            return value >= 24 && value <= 360
        case "response time":
            return value >= 1 && value <= 50
        case "gamma":
            return value >= 1 && value <= 3
        default:
            return value > 0 && value < 1000
        }
    }
    
    /// Format value with appropriate suffix
    private func formatValue(_ numberStr: String, for settingType: String) -> String {
        switch settingType.lowercased() {
        case "brightness", "contrast", "blue light filter", "text sharpness":
            return "\(numberStr)%"
        case "color temperature":
            return "\(numberStr)K"
        case "refresh rate":
            return "\(numberStr)Hz"
        case "response time":
            return "\(numberStr)ms"
        case "gamma":
            if let number = Double(numberStr) {
                return String(format: "%.1f", number)
            }
            return numberStr
        default:
            return numberStr
        }
    }
    
    /// Get debug information
    func getDebugInfo() -> (prompt: String, context: String, output: String, executionTime: Double, memoryUsage: Double) {
        return (lastPrompt, lastContext, lastOutput, lastExecutionTime, lastMemoryUsage)
    }
    
    /// Interprets a numeric LLM output in the context of a specific setting
    private func interpretSettingValue(_ output: String, settingType: String) -> String? {
        // Extract numbers from the output
        let numberPattern = try? NSRegularExpression(pattern: "\\d+\\.?\\d*")
        let range = NSRange(output.startIndex..., in: output)
        let matches = numberPattern?.matches(in: output, range: range) ?? []
        
        guard let firstMatch = matches.first,
              let numberRange = Range(firstMatch.range, in: output) else {
            // Check for ON/OFF for HDR
            if settingType.lowercased() == "hdr" {
                let lowercasedOutput = output.lowercased()
                if lowercasedOutput.contains("on") || lowercasedOutput.contains("enable") {
                    return "Enabled"
                } else if lowercasedOutput.contains("off") || lowercasedOutput.contains("disable") {
                    return "Disabled"
                }
            }
            return nil
        }
        
        let numberStr = String(output[numberRange])
        
        // Handle decimal values
        if let doubleValue = Double(numberStr) {
            // Format appropriately based on setting type
            switch settingType.lowercased() {
            case "brightness", "contrast", "blue light filter", "text sharpness":
                let intValue = Int(doubleValue)
                if intValue >= 0 && intValue <= 100 {
                    return "\(intValue)%"
                }
                
            case "refresh rate":
                let intValue = Int(doubleValue)
                if intValue >= 24 && intValue <= 360 {
                    return "\(intValue)Hz"
                }
                return "60Hz" // Fallback
                
            case "color temperature":
                let intValue = Int(doubleValue)
                if intValue < 1000 {
                    // Likely abbreviated (e.g., 65 for 6500K)
                    return "\(intValue * 100)K"
                } else if intValue >= 2000 && intValue <= 10000 {
                    return "\(intValue)K"
                }
                return "6500K" // Fallback
                
            case "response time":
                let intValue = Int(doubleValue)
                if intValue >= 1 && intValue <= 50 {
                    return "\(intValue)ms"
                }
                return "5ms" // Fallback
                
            case "gamma":
                if doubleValue >= 1.0 && doubleValue <= 3.0 {
                    return String(format: "%.1f", doubleValue)
                }
                return "2.2" // Fallback
                
            default:
                return numberStr
            }
        }
        
        return nil
    }
    
    /// Get current memory usage
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
    
    /// Converts a text prompt into token IDs for the LLM model
    private func tokenizePrompt(_ prompt: String) -> [Int32] {
        guard let vocab = loadVocab() else {
            print("❌ Failed to load vocabulary")
            return []
        }
        
        let hasSentencePiece = vocab.keys.contains { $0.hasPrefix("▁") }
        
        if hasSentencePiece {
            var tokens: [String] = []
            let words = prompt.split(separator: " ")
            
            for (i, word) in words.enumerated() {
                let token = i == 0 ? String(word) : "▁" + String(word)
                tokens.append(token)
            }
            
            let unkID = vocab["<unk>"] ?? vocab["UNK"] ?? 0
            return tokens.map { token in
                vocab[token] ?? vocab["▁" + token] ?? unkID
            }
        } else {
            let tokens = prompt.split(separator: " ").map { String($0) }
            return tokentoIDs(tokens: tokens, vocab: vocab)
        }
    }
    
    /// Helper function to convert token strings to their corresponding IDs
    private func tokentoIDs(tokens: [String], vocab: [String: Int32]) -> [Int32] {
        let unkID = vocab["<unk>"] ?? vocab["UNK"] ?? 0
        return tokens.map { vocab[$0] ?? unkID }
    }
    
    /// Converts token IDs back into human-readable text
    private func convertTokensToString(_ tokens: [Int32]) -> String {
        guard let vocab = loadVocab() else {
            return "[Error: Could not load vocabulary]"
        }
        
        return IDsToSentence(ids: tokens, vocab: vocab)
    }
    
    /// Loads the vocabulary dictionary from the app bundle
    private func loadVocab() -> [String: Int32]? {
        guard let path = Bundle.main.path(forResource: "vocab", ofType: "json"),
              let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else {
            print("❌ Failed to load vocab.json")
            return nil
        }
        do {
            let decoder = JSONDecoder()
            let vocab = try decoder.decode([String: Int32].self, from: data)
            return vocab
        } catch {
            print("❌ JSON decoding error: \(error)")
            return nil
        }
    }
    
    /// Converts token IDs back to human-readable text
    private func IDsToSentence(ids: [Int32], vocab: [String: Int32]) -> String {
        let idToToken = Dictionary(uniqueKeysWithValues: vocab.map { ($0.value, $0.key) })
        let hasSentencePiece = vocab.keys.contains { $0.hasPrefix("▁") }
        
        var result = ""
        
        for id in ids {
            guard let token = idToToken[id] else { continue }
            
            if token == "<unk>" || token == "UNK" {
                result += "[UNK]"
            }
            else if ["<s>", "</s>", "<|endoftext|>"].contains(token) {
                continue
            }
            else if token == "<0x0A>" || token == "\\n" {
                result += "\n"
            }
            else if hasSentencePiece && token.hasPrefix("▁") {
                result += " " + String(token.dropFirst())
            }
            else {
                result += token
            }
        }
        
        return result.replacingOccurrences(of: "  ", with: " ")
                   .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
