//
//  LLMtest.swift
//  On Display Computing
//
//  Created by David Zheng on 6/23/25.
//

import Foundation
import NaturalLanguage

func loadVocab() -> [String: Int32]? {
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

func tokenizeText(_ text: String, vocab: [String: Int32]) -> [Int32] {
    // This is a better tokenization approach than simple space splitting
    
    // First try to detect if this is a sentencepiece-style vocab by checking for "▁" prefix
    let hasSentencePiecePrefix = vocab.keys.contains { $0.hasPrefix("▁") }
    let unkID = vocab["<unk>"] ?? vocab["UNK"] ?? 0
    
    if hasSentencePiecePrefix {
        // SentencePiece-style tokenization - simpler approach for now
        var tokens: [String] = []
        var currentWord = ""
        
        // Add a space at the beginning to mimic SentencePiece behavior
        for (i, char) in text.enumerated() {
            if char.isWhitespace || i == 0 {
                if !currentWord.isEmpty {
                    tokens.append(currentWord)
                    currentWord = ""
                }
                if char.isWhitespace {
                    tokens.append("▁")  // Add space token
                } else {
                    currentWord += String(char)
                }
            } else {
                currentWord += String(char)
            }
        }
        
        if !currentWord.isEmpty {
            tokens.append(currentWord)
        }
        
        return tokens.map { token in
            let sentencePieceToken = token.hasPrefix("▁") ? token : "▁" + token
            return vocab[sentencePieceToken] ?? unkID
        }
    } else {
        // If not SentencePiece, use a more sophisticated approach than simple split
        var tokens = [String]()
        
        // Handle special tokens
        for specialToken in ["<s>", "</s>", "<unk>", "<pad>"] {
            if text.contains(specialToken) {
                // Simple but less efficient approach - just add special tokens separately
                tokens.append(specialToken)
            }
        }
        
        // Fallback to simpler method if no special tokens found or in addition to them
        if tokens.isEmpty {
            tokens = text.split(separator: " ").map { String($0) }
        }
        
        return tokentoIDs(tokens: tokens, vocab: vocab)
    }
}

func tokentoIDs(tokens: [String], vocab: [String: Int32]) -> [Int32] {
    let unkID = vocab["<unk>"] ?? vocab["UNK"] ?? 0
    return tokens.map { vocab[$0] ?? unkID }
}

func IDsToSentence(ids: [Int32], vocab: [String: Int32]) -> String {
    let idToToken = Dictionary(uniqueKeysWithValues: vocab.map { ($0.value, $0.key) })
    
    var result = ""
    var skipNextSpace = false
    
    for id in ids {
        guard let token = idToToken[id] else { continue }
        
        switch token {
        case "</s>", "<s>", "<|endoftext|>":
            skipNextSpace = true  // Skip adding space after special tokens
            continue  // Skip these tokens entirely
        case "<0x0A>", "\\n":
            result += "\n"
            skipNextSpace = true
        case token where token.hasPrefix("<|") && token.hasSuffix("|>"):
            result += "\n\n[" + token.dropFirst(2).dropLast(2) + "]: "
            skipNextSpace = true
        case token where token.hasPrefix("▁"):
            // For SentencePiece tokens that represent space+characters
            result += " " + token.dropFirst()
        default:
            if token.hasPrefix("▁") {
                result += " " + String(token.dropFirst())
            } else if !skipNextSpace {
                result += " " + token
            } else {
                result += token
            }
        }
        skipNextSpace = false
    }
    
    return result.trimmingCharacters(in: .whitespacesAndNewlines)
}

func readableToken(_ token: String) -> String {
    switch token {
    case "\n": return "[NEWLINE]"
    case "\t": return "[TAB]"
    case " ":  return "[SPACE]"
    default:
        if token.hasPrefix("▁") {
            // Handle SentencePiece tokens by replacing the prefix with space
            return " " + token.dropFirst()
        }
        return token
    }
}

func LLMtest() {
    guard let engine = LLMEngine() else {
        print("❌ LLMEngine failed to initialize")
        return
    }
    
    guard let vocab = loadVocab() else {
        print("❌ Vocab file not loaded")
        return
    }
    
    print("📚 Vocab size: \(vocab.count)")
    
    // Check if important words are in vocabulary
    let checkWords = ["capital", "France", "Beijing", "is", "the", "of", "?"]
    print("\n📋 Checking if important words are in vocabulary:")
    for word in checkWords {
        if let id = vocab[word] {
            print("  ✅ \"\(word)\" → ID: \(id)")
        } else if let id = vocab["▁" + word] {
            print("  ✅ \"\(word)\" → ID: \(id) (as ▁\(word))")
        } else {
            print("  ❌ \"\(word)\" NOT FOUND in vocabulary!")
        }
    }
    
    // Determine if it's a SentencePiece vocabulary
    let hasSentencePiece = vocab.keys.contains(where: { $0.hasPrefix("▁") })
    print("\n🔤 Vocabulary type: \(hasSentencePiece ? "SentencePiece (▁ prefix)" : "Standard")")
    
    // Try different prompt options
    let promptOptions = [
        "The capital of China is"
    ]
    
    let selectedPromptIndex = 0  // Try option #2
    let selectedPrompt = promptOptions[selectedPromptIndex]
    print("\n📝 Using prompt: \"\(selectedPrompt)\"")
    
    // Tokenize appropriately based on vocabulary type
    var inputIDs: [Int32] = []
    if hasSentencePiece {
        // Basic SentencePiece-style tokenization
        var tokens: [String] = []
        for (i, word) in selectedPrompt.split(separator: " ").enumerated() {
            // Add space prefix to all words except the first one
            let token = i == 0 ? String(word) : "▁" + String(word)
            tokens.append(token)
        }
        inputIDs = tokens.compactMap {
            vocab[$0] ?? vocab["▁" + $0] ?? vocab["<unk>"] ?? vocab["UNK"]
        }
    } else {
        // Standard tokenization
        let tokens = selectedPrompt.split(separator: " ").map { String($0) }
        inputIDs = tokentoIDs(tokens: tokens, vocab: vocab)
    }
    
    print("🔢 Tokenized to: \(inputIDs)")
    
    // Temperature (0.0 = greedy, higher = more random)
    let temperature: Float = 0.0
    
    // Generate with selected parameters
    print("\n🚀 Starting generation (temperature: \(temperature))...")
    let generatedIDs = engine.generateStreaming(
        inputIDs: inputIDs,
        maxTokens: 20,
        eosTokenID: vocab["</s>"],
        temperature: temperature
    )
    
    print("🧩 Generated \(generatedIDs.count) tokens")
    
    let fullOutput = IDsToSentence(ids: inputIDs + generatedIDs, vocab: vocab)
    print("\n📝 Full output: \(fullOutput)")
}
