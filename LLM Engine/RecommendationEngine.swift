import Foundation

/// Generates display setting recommendations using the LLM engine
class RecommendationEngine {
    /// The underlying machine learning model for generating recommendations
    private let llmEngine: LLMEngine
    
    /// Dictionary to cache recommendations for apps we've already seen
    private var recommendationCache: [String: String] = [:]
    
    /// For debugging
    private var lastPrompt: String = ""
    private var lastOutput: String = ""
    
    /// Initializes the recommendation engine with an LLM model
    init?() {
        // Try to create the LLM engine, fail initialization if that fails
        guard let engine = LLMEngine() else { return nil }
        self.llmEngine = engine
        print("🚀 RecommendationEngine initialized with LLMEngine")
    }
    
    /// Gets display setting recommendations for a specific application
    /// - Parameters:
    ///   - appName: Name of the application
    ///   - appType: Type/category of the application
    /// - Returns: String containing recommended display settings
    func getRecommendations(for appName: String, appType: String) -> String {
        print("⚙️ Getting recommendations for \(appName) (\(appType))")
        
        // Create a cache key from app name and type
        let cacheKey = "\(appName)-\(appType)"
        
        // Check if we already have cached recommendations for this app
        if let cachedRecommendations = recommendationCache[cacheKey] {
            print("📋 Using cached recommendations for \(appName)")
            return cachedRecommendations
        }
        
        // First try with completion-style prompts for specific settings
        var settings = [String: String]()
        
        // Try to get brightness
        if let brightness = getSpecificSetting("For \(appType), the brightness setting should be", suffix: "%") {
            settings["Brightness"] = brightness + "%"
        }
        
        // Try to get contrast
        if let contrast = getSpecificSetting("For \(appType), the contrast setting should be", suffix: "%") {
            settings["Contrast"] = contrast + "%"
        }
        
        // Try to get color temperature
        if let colorTemp = getSpecificSetting("For \(appType), the color temperature setting should be", suffix: "K") {
            settings["Color Temperature"] = colorTemp + "K"
        }
        
        if let gamma = getSpecificSetting("For \(appType), refresh rate should be") {
            settings["Gamma"] = gamma
        }
        if let blue = getSpecificSetting("For \(appType), the Blue Light Filter should be", suffix: "%") {
            settings["Blue Light Filter"] = blue + "%"
            }
        if let responseTime = getSpecificSetting("For \(appType), the response time should be", suffix: "ms") {
            settings["Response Time"] = responseTime + "ms"
            }
        if let refreshRate = getSpecificSetting("For \(appType), the refresh rate should be", suffix: "Hz") {
            settings["Refresh Rate"] = refreshRate + "Hz"
            }
        if let textSharpness = getSpecificSetting("For \(appType), the text sharpness should be", suffix: "%") {
            settings["Text Sharpness"] = textSharpness + "%"
            }
        if let hdr = getSpecificSetting("For \(appType), the hdr should be ") {
            settings["HDR"] = hdr
            }
        // Generate full result
        var result = ""
        for (setting, value) in settings {
            result += "\(setting): \(value)\n"
        }
        
        // If we got at least one setting, use that, otherwise use defaults
        if !result.isEmpty {
            // Cache the result
            recommendationCache[cacheKey] = result
            return result
        } else {
            print("⚠️ Could not get any settings from LLM, using defaults")
            let defaults = getDefaultSettings(for: appType)
            recommendationCache[cacheKey] = defaults
            return defaults
        }
    }
    
    /// Get a specific setting using direct prompting
    private func getSpecificSetting(_ prompt: String, suffix: String = "") -> String? {
        self.lastPrompt = prompt
        
        // Convert the text prompt to token IDs
        let tokens = tokenizePrompt(prompt)
        
        // Generate with a low temperature for more deterministic results
        let generatedIDs = llmEngine.generateStreaming(
            inputIDs: tokens,
            maxTokens: 10,  // We only need a short response
            temperature: 0.0
        )
        
        // Convert back to text
        if !generatedIDs.isEmpty {
            let output = convertTokensToString(generatedIDs)
            self.lastOutput = output
            print("📝 Raw LLM output: \(output)")
            
            // Try to extract a number
            if let numberRange = output.range(of: "\\d+", options: .regularExpression) {
                let numberStr = String(output[numberRange])
                if let number = Int(numberStr), number > 0 && number < 1000 {
                    // Looks like a valid value
                    return numberStr
                }
            }
        }
        
        return nil
    }
    
    /// Returns the last prompt and output for debugging
    func getDebugInfo() -> (prompt: String, output: String) {
        return (lastPrompt, lastOutput)
    }
    
    /// Provides default display settings if LLM generation fails
    private func getDefaultSettings(for appType: String) -> String {
        switch appType {
        case "photo editing app":
            return """
            Brightness: 85%
            Contrast: 80%
            Color Temperature: 6500K
            Gamma: 2.2
            """
            
        case "video editing app":
            return """
            Brightness: 90%
            Contrast: 85%
            Color Temperature: 6000K
            Response Time: 5ms
            """
            
        case "text editing app":
            return """
            Brightness: 65%
            Blue Light Filter: 40%
            Text Sharpness: 75%
            Color Temperature: 5500K
            """
            
        case "code editor":
            return """
            Brightness: 60%
            Blue Light Filter: 50%
            Contrast: 70%
            Text Sharpness: 80%
            """
            
        case "gaming app":
            return """
            Brightness: 90%
            Contrast: 85%
            Response Time: 1ms
            HDR: Enabled
            """
            
        case "web browser":
            return """
            Brightness: 70%
            Blue Light Filter: 30%
            Contrast: 75%
            Color Temperature: 5800K
            """
            
        case "terminal app":
            return """
            Brightness: 55%
            Contrast: 80%
            Text Sharpness: 85%
            Blue Light Filter: 45%
            """
            
        default:
            return """
            Brightness: 75%
            Contrast: 75%
            Color Temperature: 6000K
            Blue Light Filter: 35%
            """
        }
    }
    
    /// Converts a text prompt into token IDs for the LLM model
    private func tokenizePrompt(_ prompt: String) -> [Int32] {
        // Get the vocabulary for the model
        guard let vocab = loadVocab() else {
            print("❌ Failed to load vocabulary")
            return []
        }
        
        // Check if the vocabulary uses SentencePiece format (with ▁ prefix)
        let hasSentencePiece = vocab.keys.contains { $0.hasPrefix("▁") }
        
        // Tokenize differently based on vocabulary type
        if hasSentencePiece {
            // Handle SentencePiece tokenization
            var tokens: [String] = []
            let words = prompt.split(separator: " ")
            
            for (i, word) in words.enumerated() {
                // Add space prefix to all words except the first one
                let token = i == 0 ? String(word) : "▁" + String(word)
                tokens.append(token)
            }
            
            // Convert tokens to their corresponding IDs
            let unkID = vocab["<unk>"] ?? vocab["UNK"] ?? 0
            return tokens.map { token in
                vocab[token] ?? vocab["▁" + token] ?? unkID
            }
        } else {
            // Standard tokenization (splitting by spaces)
            let tokens = prompt.split(separator: " ").map { String($0) }
            return tokentoIDs(tokens: tokens, vocab: vocab)
        }
    }
    
    /// Converts token IDs back into human-readable text
    /// - Parameter tokens: Array of token IDs from the model
    /// - Returns: String representation of the tokens
    private func convertTokensToString(_ tokens: [Int32]) -> String {
        // Get the vocabulary for the model
        guard let vocab = loadVocab() else {
            return "[Error: Could not load vocabulary]"
        }
        
        // Convert the list of IDs to a string
        return IDsToSentence(ids: tokens, vocab: vocab)
    }
    
    /// Helper function to convert token strings to their corresponding IDs
    /// - Parameters:
    ///   - tokens: Array of token strings
    ///   - vocab: Dictionary mapping tokens to their IDs
    /// - Returns: Array of token IDs
    private func tokentoIDs(tokens: [String], vocab: [String: Int32]) -> [Int32] {
        let unkID = vocab["<unk>"] ?? vocab["UNK"] ?? 0
        return tokens.map { vocab[$0] ?? unkID }
    }
    
    /// Loads the vocabulary dictionary from the app bundle
    /// - Returns: Dictionary mapping token strings to their IDs
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
    /// - Parameters:
    ///   - ids: Array of token IDs
    ///   - vocab: Dictionary mapping tokens to their IDs
    /// - Returns: Human-readable string representation
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
                // Skip these special tokens
                continue
            }
            else if token == "<0x0A>" || token == "\\n" {
                result += "\n"
            }
            else if hasSentencePiece && token.hasPrefix("▁") {
                // For SentencePiece tokens, replace ▁ with space
                result += " " + String(token.dropFirst())
            }
            else {
                result += token
            }
        }
        
        // Clean up spacing
        return result.replacingOccurrences(of: "  ", with: " ")
                   .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
