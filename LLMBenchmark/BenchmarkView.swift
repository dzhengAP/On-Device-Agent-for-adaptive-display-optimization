import SwiftUI

struct BenchmarkView: View {
    let viewModel: AppViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var isRunning = false
    @State private var results: [String] = []
    
    var body: some View {
        VStack(spacing: 20) {
            // Header
            HStack {
                Text("DisplayAgent Performance Benchmark")
                    .font(.largeTitle)
                    .fontWeight(.bold)
                
                Spacer()
                
                Button("Done") {
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
            }
            .padding()
            
            Divider()
            
            // Content
            if isRunning {
                VStack(spacing: 20) {
                    ProgressView()
                        .scaleEffect(1.5)
                    
                    Text("Running LLM inference benchmarks...")
                        .font(.headline)
                    
                    Text("Testing all acceleration methods (CPU, CoreML, Metal, MPS)")
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding(40)
            } else if results.isEmpty {
                VStack(spacing: 20) {
                    Text("Ready to benchmark LLM inference performance")
                        .font(.title2)
                        .multilineTextAlignment(.center)
                    
                    Text("This will test inference speed across different acceleration methods")
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                    
                    Button(action: startBenchmark) {
                        HStack {
                            Image(systemName: "bolt.fill")
                            Text("Start Inference Benchmark")
                        }
                        .font(.headline)
                        .padding()
                        .frame(maxWidth: 300)
                        .background(Color.yellow)
                        .foregroundColor(.black)
                        .cornerRadius(12)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding(40)
            } else {
                VStack(spacing: 15) {
                    Text("Benchmark Results")
                        .font(.title2)
                        .fontWeight(.semibold)
                    
                    ScrollView {
                        VStack(spacing: 10) {
                            ForEach(results.indices, id: \.self) { index in
                                Text(results[index])
                                    .font(.system(.body, design: .monospaced))
                                    .padding()
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .background(Color.gray.opacity(0.1))
                                    .cornerRadius(8)
                            }
                        }
                        .padding()
                    }
                    
                    HStack(spacing: 15) {
                        Button(action: startBenchmark) {
                            HStack {
                                Image(systemName: "arrow.clockwise")
                                Text("Run Again")
                            }
                            .font(.headline)
                            .padding()
                            .frame(maxWidth: 200)
                            .background(Color.green)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                        }
                        
                        Button(action: copyResults) {
                            HStack {
                                Image(systemName: "doc.on.doc")
                                Text("Copy Results")
                            }
                            .font(.headline)
                            .padding()
                            .frame(maxWidth: 200)
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                        }
                    }
                }
                .padding()
            }
            
            Spacer()
        }
        .frame(width: 700, height: 600)
        .background(Color(NSColor.windowBackgroundColor))
    }
    
    private func startBenchmark() {
        print("🔧 Checking LLM engine availability...")
        
        if let engine = viewModel.getLLMEngine() {
            print("✅ LLM engine found: \(engine)")
            // Run the actual benchmark
            isRunning = true
            results = []
            
            DispatchQueue.global().async {
                // Run our custom inference benchmark
                runQuickInferenceBenchmark()
                
                // Wait a moment for console output to complete
                DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                    self.results = [
                        "🚀 DisplayAgent Inference Benchmark Complete",
                        "",
                        "Check Xcode console for detailed results including:",
                        "• Per-method timing breakdown",
                        "• Memory usage comparison",
                        "• Tokens per second metrics",
                        "• Generated text samples",
                        "",
                        "Results also saved to system logs.",
                        "",
                        "To view results: Open Xcode > View > Debug Area > Console"
                    ]
                    self.isRunning = false
                }
            }
        } else {
            print("❌ LLM engine is nil")
            
            // Check if it's still initializing
            if viewModel.recommendations == "Waiting for app detection..." {
                results = ["⏳ LLM engine is still initializing. Please wait and try again."]
            } else {
                results = ["❌ No LLM engine available for benchmarking"]
            }
        }
    }
    
    private func copyResults() {
        let resultText = results.joined(separator: "\n")
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(resultText, forType: .string)
    }
}
