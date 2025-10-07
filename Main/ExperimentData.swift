//
//  ExperimentData.swift
//  On Display Computing
//

import SwiftUI

/// Container for real experiment results (no simulated data).
final class ExperimentData: ObservableObject {

    @Published var accuracyResults: [LLMBenchmark.AccuracyResult] = []
    @Published var performanceResults: [LLMBenchmark.PerformanceResult] = []
    @Published var scalabilityResults: [LLMBenchmark.ScalabilityResult] = []
    @Published var userExperienceMetrics: [LLMBenchmark.UserExperienceMetric] = []

    @Published var isRunning = false
    @Published var experimentProgress = ""

    private let benchmark = LLMBenchmark.shared

    // MARK: - Runner

    func runAllExperiments(viewModel: AppViewModel) {
        guard !isRunning else { return }
        isRunning = true
        experimentProgress = "Starting experiments…"

        let group = DispatchGroup()

        // Accuracy
        group.enter()
        experimentProgress = "Running accuracy evaluation…"
        benchmark.runAccuracyEvaluation(viewModel: viewModel) { results in
            DispatchQueue.main.async {
                self.accuracyResults = results
                group.leave()
            }
        }

        // Performance
        if let engine = viewModel.getLLMEngine() ?? LLMEngine() {
            group.enter()
            DispatchQueue.main.async { self.experimentProgress = "Running performance benchmarking…" }
            benchmark.runPerformanceBenchmarking(engine: engine) { results in
                DispatchQueue.main.async {
                    self.performanceResults = results
                    group.leave()
                }
            }
        }

        // Scalability
        group.enter()
        DispatchQueue.main.async { self.experimentProgress = "Running scalability testing…" }
        benchmark.runScalabilityTesting(viewModel: viewModel) { results in
            DispatchQueue.main.async {
                self.scalabilityResults = results
                group.leave()
            }
        }

        // UX
        group.enter()
        DispatchQueue.main.async { self.experimentProgress = "Running user experience analysis…" }
        benchmark.runUserExperienceAnalysis { metrics in
            DispatchQueue.main.async {
                self.userExperienceMetrics = metrics
                group.leave()
            }
        }

        group.notify(queue: .main) {
            self.isRunning = false
            self.experimentProgress = "Experiments completed"
        }
    }

    // MARK: - Aggregations (computed from real results)

    /// Average accuracy per engine type (percent)
    var aggregatedAccuracyData: [(String, Double, Color)] {
        let grouped = Dictionary(grouping: accuracyResults, by: { $0.engineType })
        return grouped.compactMap { engine, rows in
            guard !rows.isEmpty else { return nil }
            let avg = rows.reduce(0.0, { $0 + $1.accuracy }) / Double(rows.count)
            let color: Color = {
                switch engine {
                case .llmOnly: return .blue
                case .ragOnly: return .green
                case .hybrid:  return .purple
                }
            }()
            return (engine.rawValue, avg * 100.0, color)
        }.sorted { $0.1 < $1.1 }
    }

    /// Human-readable performance takeaways from real results.
    var performanceInsights: [String] {
        guard !performanceResults.isEmpty else { return ["No performance data available"] }
        var insights: [String] = []

        if let bestLatency = performanceResults.min(by: { $0.averageLatencyMs < $1.averageLatencyMs }),
           let worstLatency = performanceResults.max(by: { $0.averageLatencyMs < $1.averageLatencyMs }) {
            let delta = ((worstLatency.averageLatencyMs - bestLatency.averageLatencyMs) / worstLatency.averageLatencyMs) * 100
            insights.append("• \(bestLatency.method.rawValue) reduces latency by \(String(format: "%.1f", delta))% vs \(worstLatency.method.rawValue)")
        }

        if let bestThroughput = performanceResults.max(by: { $0.throughputTokensPerSec < $1.throughputTokensPerSec }) {
            insights.append("• Peak throughput: \(String(format: "%.1f", bestThroughput.throughputTokensPerSec)) tokens/s with \(bestThroughput.method.rawValue)")
        }

        if let memBest = performanceResults.min(by: { $0.memoryUsageMB < $1.memoryUsageMB }) {
            insights.append("• Most memory-efficient: \(memBest.method.rawValue) (\(String(format: "%.1f", memBest.memoryUsageMB))MB)")
        }

        if let powerBest = performanceResults.min(by: { $0.powerConsumptionW < $1.powerConsumptionW }) {
            insights.append("• Lowest power draw: \(powerBest.method.rawValue) (\(String(format: "%.1f", powerBest.powerConsumptionW))W)")
        }

        return insights
    }

    /// Simple ablation-style insights comparing engines (from real accuracy).
    var ablationInsights: [String] {
        let rag = accuracyResults.filter { $0.engineType == .ragOnly }
        let llm = accuracyResults.filter { $0.engineType == .llmOnly }
        let hybrid = accuracyResults.filter { $0.engineType == .hybrid }

        var out: [String] = []

        if !rag.isEmpty && !llm.isEmpty {
            let ragAvg = rag.reduce(0.0, { $0 + $1.accuracy }) / Double(rag.count)
            let llmAvg = llm.reduce(0.0, { $0 + $1.accuracy }) / Double(llm.count)
            let lift = ((ragAvg - llmAvg) / max(llmAvg, .leastNonzeroMagnitude)) * 100
            out.append("RAG Contribution:")
            out.append("• +\(String(format: "%.1f", lift))% vs LLM-only")
        }

        if !hybrid.isEmpty && !rag.isEmpty {
            let hybridAvg = hybrid.reduce(0.0, { $0 + $1.accuracy }) / Double(hybrid.count)
            let ragAvg = rag.reduce(0.0, { $0 + $1.accuracy }) / Double(rag.count)
            let lift = ((hybridAvg - ragAvg) / max(ragAvg, .leastNonzeroMagnitude)) * 100
            out.append("LLM Contribution in Hybrid:")
            out.append("• +\(String(format: "%.1f", lift))% vs RAG-only")
        }

        return out.isEmpty ? ["Insufficient data for ablation analysis"] : out
    }
}

// MARK: - Simple chart primitives (displaying ONLY real data)

struct SimpleBarChart: View {
    let data: [(String, Double, Color)]
    let title: String
    let maxValue: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title).font(.headline).fontWeight(.semibold)

            if data.isEmpty {
                Text("No data available – run experiments to see results")
                    .font(.caption).foregroundColor(.gray).padding()
            } else {
                VStack(spacing: 8) {
                    ForEach(0..<data.count, id: \.self) { i in
                        let item = data[i]
                        HStack {
                            Text(item.0)
                                .frame(width: 120, alignment: .leading)
                                .font(.caption)

                            GeometryReader { geo in
                                HStack(spacing: 0) {
                                    Rectangle()
                                        .fill(item.2)
                                        .frame(width: geo.size.width * (max(0, item.1) / max(1, maxValue)))
                                    Spacer()
                                }
                            }
                            .frame(height: 20)

                            Text(String(format: "%.1f", item.1))
                                .frame(width: 50, alignment: .trailing)
                                .font(.caption)
                        }
                    }
                }
            }
        }
        .padding()
    }
}

// MARK: - Real-data views

// Updated sections of ExperimentData.swift with DisplayAgent branding

struct SimpleAccuracyChartView: View {
    @ObservedObject var experimentData: ExperimentData

    var body: some View {
        VStack(spacing: 20) {
            Text("DisplayAgent: Accuracy Comparison")
                .font(.title).fontWeight(.bold).padding(.top)

            if experimentData.isRunning {
                VStack(spacing: 10) {
                    ProgressView()
                    Text(experimentData.experimentProgress).font(.caption)
                }
                .frame(height: 200)
            } else {
                SimpleBarChart(
                    data: experimentData.aggregatedAccuracyData,
                    title: "Recommendation Accuracy by Engine Type",
                    maxValue: 100.0
                )
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Key Findings").font(.headline).padding(.bottom, 4)
                if experimentData.isRunning {
                    Text("Running experiments…").font(.subheadline).foregroundColor(.gray)
                } else if experimentData.accuracyResults.isEmpty {
                    Text("No experimental data available. Run experiments to collect results.")
                        .font(.subheadline).foregroundColor(.gray)
                } else {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(experimentData.ablationInsights, id: \.self) { s in
                            Text(s).font(.subheadline)
                        }
                        let n = experimentData.accuracyResults.count
                        let avgT = experimentData.accuracyResults.reduce(0.0, { $0 + $1.executionTimeMs }) / Double(n)
                        Text("• Average response time: \(String(format: "%.1f", avgT)) ms").font(.subheadline)
                        Text("• Total test cases: \(n)").font(.subheadline)
                    }
                }
            }
            .padding()
            .background(Color.gray.opacity(0.1))
            .cornerRadius(8)

            Spacer()
        }
        .padding()
        .frame(width: 800, height: 600)
        .background(Color.white)
    }
}

struct SimplePerformanceChartView: View {
    @ObservedObject var experimentData: ExperimentData

    private var latencyData: [(String, Double, Color)] {
        experimentData.performanceResults.map { r in
            let c: Color = {
                switch r.method {
                case .cpuOnly: return .red
                case .coreMLDefault: return .orange
                case .metalOptimized: return .blue
                case .mps: return .green
                }
            }()
            return (r.method.rawValue, r.averageLatencyMs, c)
        }
    }

    private var throughputData: [(String, Double, Color)] {
        experimentData.performanceResults.map { r in
            let c: Color = {
                switch r.method {
                case .cpuOnly: return .red
                case .coreMLDefault: return .orange
                case .metalOptimized: return .blue
                case .mps: return .green
                }
            }()
            return (r.method.rawValue, r.throughputTokensPerSec, c)
        }
    }

    var body: some View {
        VStack(spacing: 20) {
            Text("DisplayAgent: Performance Analysis")
                .font(.title).fontWeight(.bold).padding(.top)

            if experimentData.isRunning {
                VStack(spacing: 10) {
                    ProgressView()
                    Text(experimentData.experimentProgress).font(.caption)
                }
                .frame(height: 300)
            } else {
                HStack(spacing: 40) {
                    SimpleBarChart(
                        data: latencyData,
                        title: "Average Latency (ms)",
                        maxValue: latencyData.map(\.1).max() ?? 1.0
                    )

                    SimpleBarChart(
                        data: throughputData,
                        title: "Throughput (tokens/s)",
                        maxValue: throughputData.map(\.1).max() ?? 1.0
                    )
                }
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Performance Insights").font(.headline).padding(.bottom, 4)
                if experimentData.isRunning {
                    Text("Collecting performance data…").font(.subheadline).foregroundColor(.gray)
                } else if experimentData.performanceResults.isEmpty {
                    Text("No performance data available. Run experiments to see results.")
                        .font(.subheadline).foregroundColor(.gray)
                } else {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(experimentData.performanceInsights, id: \.self) { s in
                            Text(s).font(.subheadline)
                        }
                    }
                }
            }
            .padding()
            .background(Color.green.opacity(0.1))
            .cornerRadius(8)

            Spacer()
        }
        .padding()
        .frame(width: 800, height: 600)
        .background(Color.white)
    }
}

struct SimpleAblationChartView: View {
    @ObservedObject var experimentData: ExperimentData

    var body: some View {
        VStack(spacing: 20) {
            Text("DisplayAgent: Ablation Study")
                .font(.title).fontWeight(.bold).padding(.top)

            if experimentData.isRunning {
                VStack(spacing: 10) {
                    ProgressView()
                    Text(experimentData.experimentProgress).font(.caption)
                }
                .frame(height: 200)
            } else {
                SimpleBarChart(
                    data: experimentData.aggregatedAccuracyData,
                    title: "Component Contribution (accuracy, %)",
                    maxValue: 100.0
                )
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Ablation Analysis").font(.headline).padding(.bottom, 4)
                if experimentData.isRunning {
                    Text("Analyzing component contributions…").font(.subheadline).foregroundColor(.gray)
                } else if experimentData.accuracyResults.isEmpty {
                    Text("No ablation data available. Run experiments to analyze components.")
                        .font(.subheadline).foregroundColor(.gray)
                } else {
                    ForEach(experimentData.ablationInsights, id: \.self) { s in
                        Text(s).font(.subheadline)
                    }
                }
            }
            .padding()

            Spacer()
        }
        .padding()
        .frame(width: 800, height: 600)
        .background(Color.white)
    }
}

struct ExperimentRunnerView: View {
    @StateObject private var experimentData = ExperimentData()
    let viewModel: AppViewModel

    var body: some View {
        VStack(spacing: 20) {
            HStack {
                Text("DisplayAgent: LLM Experiments")
                    .font(.title).fontWeight(.bold)
                Spacer()
                Button {
                    experimentData.runAllExperiments(viewModel: viewModel)
                } label: {
                    HStack {
                        if experimentData.isRunning {
                            ProgressView().scaleEffect(0.8)
                            Text("Running…")
                        } else {
                            Image(systemName: "play.fill")
                            Text("Run Experiments")
                        }
                    }
                }
                .disabled(experimentData.isRunning)
                .buttonStyle(.borderedProminent)
            }

            TabView {
                SimpleAccuracyChartView(experimentData: experimentData)
                    .tabItem { Label("Accuracy", systemImage: "target") }

                SimplePerformanceChartView(experimentData: experimentData)
                    .tabItem { Label("Performance", systemImage: "speedometer") }

                SimpleAblationChartView(experimentData: experimentData)
                    .tabItem { Label("Ablation", systemImage: "chart.bar.xaxis") }
            }
        }
        .padding()
    }
}

struct SimpleScalabilityChartView: View {
    @ObservedObject var experimentData: ExperimentData

    private var throughputData: [(String, Double, Color)] {
        experimentData.scalabilityResults.map { r in
            ("\(r.concurrencyLevel)x", r.throughputRequestsPerSecond, .orange)
        }
    }

    var body: some View {
        VStack(spacing: 20) {
            Text("DisplayAgent: Scalability")
                .font(.title)
                .fontWeight(.bold)
                .padding(.top)

            if experimentData.isRunning {
                VStack(spacing: 10) {
                    ProgressView()
                    Text(experimentData.experimentProgress).font(.caption)
                }
                .frame(height: 200)
            } else {
                SimpleBarChart(
                    data: throughputData,
                    title: "Throughput (req/s) by Concurrency",
                    maxValue: max(throughputData.map(\.1).max() ?? 1.0, 1.0)
                )
            }

            // A couple of quick scalabilty stats
            if !experimentData.scalabilityResults.isEmpty {
                let best = experimentData.scalabilityResults.max(by: { $0.throughputRequestsPerSecond < $1.throughputRequestsPerSecond })
                let avgSuccess = experimentData.scalabilityResults.map(\.successRate).reduce(0, +) / Double(experimentData.scalabilityResults.count)

                VStack(alignment: .leading, spacing: 6) {
                    if let best {
                        Text("• Peak throughput at \(best.concurrencyLevel)x: \(String(format: "%.2f", best.throughputRequestsPerSecond)) req/s")
                            .font(.subheadline)
                    }
                    Text("• Avg success rate: \(String(format: "%.1f%%", avgSuccess * 100))")
                        .font(.subheadline)
                }
                .padding()
                .background(Color.orange.opacity(0.08))
                .cornerRadius(8)
            }

            Spacer()
        }
        .padding()
        .frame(width: 800, height: 600)
        .background(Color.white)
    }
}
