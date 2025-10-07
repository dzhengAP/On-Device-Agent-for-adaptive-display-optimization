import SwiftUI

// MARK: - ContentView

struct ContentView: View {
    @ObservedObject var viewModel = AppViewModel()
    @StateObject var experimentData = ExperimentData()

    // Predeclare grid columns so the type checker doesn't re-infer a giant expression
    private let experimentColumns: [GridItem] = [
        GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible())
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            HeaderBar(viewModel: viewModel, experimentData: experimentData)

            RecommendationsPanel(viewModel: viewModel)

            ExperimentsPanel(
                experimentData: experimentData,
                columns: experimentColumns,
                runAll: { experimentData.runAllExperiments(viewModel: viewModel) },
                runAccuracy: runRealAccuracyExperiment,
                runPerformance: runRealPerformanceExperiment,
                runAblation: showRealAblationResults,
                runScalability: runRealScalabilityExperiment,
                runUX: runRealUXExperiment,
                showCharts: { viewModel.showRealChartsView() },
                showBenchmark: { viewModel.showBenchmarkView() },
                exportJSON: exportRealExperimentData,
                exportCharts: exportRealCharts,
                exportReport: exportAnalysisReport,
                exportAll: exportAllRealData
            )

            if !experimentData.accuracyResults.isEmpty || !experimentData.performanceResults.isEmpty {
                RealExperimentStatsView(experimentData: experimentData)
            }

            ResultsSummaryPanel(experimentData: experimentData)

            Spacer()
        }
        .padding()
        .frame(width: 700, height: 800)
        .sheet(isPresented: $viewModel.isBenchmarkViewVisible) {
            BenchmarkView(viewModel: viewModel)
        }
        .sheet(isPresented: $viewModel.isRealChartsViewVisible) {
            ExperimentRunnerView(viewModel: viewModel)   // uses its own @StateObject internally
        }
    }
}

// MARK: - Header

private struct HeaderBar: View {
    @ObservedObject var viewModel: AppViewModel
    @ObservedObject var experimentData: ExperimentData

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("DisplayAgent: Adaptive Display Intelligence")
                    .font(.largeTitle).fontWeight(.bold)
                Spacer()
                HStack(spacing: 8) {
                    Circle()
                        .fill(experimentData.isRunning ? Color.orange : Color.green)
                        .frame(width: 12, height: 12)
                    Text(experimentData.isRunning ? experimentData.experimentProgress : "Ready")
                        .font(.caption).foregroundColor(.secondary)
                }
            }

            HStack {
                Text("Active Application:").fontWeight(.semibold)
                Text(viewModel.currentApp).foregroundColor(.blue)
                Spacer()
                Text("Type:").fontWeight(.semibold)
                Text(viewModel.appType).foregroundColor(.blue)
            }
        }
    }
}

// MARK: - Recommendations

private struct RecommendationsPanel: View {
    @ObservedObject var viewModel: AppViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Current Display Recommendations")
                .font(.title2).fontWeight(.semibold)

            ScrollView {
                Text(viewModel.recommendations)
                    .padding()
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .lineSpacing(8)
            }
            .frame(height: 100)
            .background(Color.gray.opacity(0.1))
            .cornerRadius(8)

            if let settings = viewModel.parsedSettings {
                SettingsView(settings: settings)
            }

            HStack {
                Toggle("Auto-apply settings", isOn: $viewModel.autoApplySettings)
                Spacer()
                Button("Apply Now") { viewModel.applyCurrentSettings() }
                    .disabled(viewModel.parsedSettings == nil)
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding(.top, 4)
        .overlay(Divider().padding(.top), alignment: .topLeading)
    }
}

// MARK: - Experiments Panel

private struct ExperimentsPanel: View {
    @ObservedObject var experimentData: ExperimentData
    let columns: [GridItem]

    let runAll: () -> Void
    let runAccuracy: () -> Void
    let runPerformance: () -> Void
    let runAblation: () -> Void
    let runScalability: () -> Void
    let runUX: () -> Void
    let showCharts: () -> Void
    let showBenchmark: () -> Void  // NEW: Added benchmark function

    let exportJSON: () -> Void
    let exportCharts: () -> Void
    let exportReport: () -> Void
    let exportAll: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            HStack {
                Text("DisplayAgent Experiments")
                    .font(.title2).fontWeight(.bold)
                Spacer()
                if experimentData.isRunning {
                    ProgressView().scaleEffect(0.8)
                    Text("Running Real Experiments…")
                        .font(.caption).foregroundColor(.orange)
                }
            }

            LazyVGrid(columns: columns, spacing: 12) {
                ExperimentButton(
                    title: "Accuracy\nEvaluation",
                    icon: "target",
                    color: .green,
                    description: "Real LLM vs RAG vs Hybrid testing",
                    badge: experimentData.accuracyResults.isEmpty ? nil : "\(experimentData.accuracyResults.count)",
                    action: runAccuracy
                )
                .disabled(experimentData.isRunning)

                ExperimentButton(
                    title: "Performance\nProfiling",
                    icon: "speedometer",
                    color: .blue,
                    description: "CPU/GPU/MPS acceleration testing",
                    badge: experimentData.performanceResults.isEmpty ? nil : "\(experimentData.performanceResults.count)",
                    action: runPerformance
                )
                .disabled(experimentData.isRunning)

                ExperimentButton(
                    title: "LLM Inference\nBenchmark",
                    icon: "bolt.fill",
                    color: .yellow,
                    description: "Quick inference speed comparison",
                    badge: nil,
                    action: showBenchmark
                )
                .disabled(experimentData.isRunning)

                ExperimentButton(
                    title: "Ablation\nStudy",
                    icon: "flask",
                    color: .purple,
                    description: "Component contribution analysis",
                    badge: nil,
                    action: runAblation
                )
                .disabled(experimentData.isRunning)

                ExperimentButton(
                    title: "Scalability\nAnalysis",
                    icon: "chart.line.uptrend.xyaxis",
                    color: .orange,
                    description: "Concurrent load testing",
                    badge: experimentData.scalabilityResults.isEmpty ? nil : "\(experimentData.scalabilityResults.count)",
                    action: runScalability
                )
                .disabled(experimentData.isRunning)

                ExperimentButton(
                    title: "User Experience\nMetrics",
                    icon: "person.2",
                    color: .pink,
                    description: "UX performance indicators",
                    badge: experimentData.userExperienceMetrics.isEmpty ? nil : "\(experimentData.userExperienceMetrics.count)",
                    action: runUX
                )
                .disabled(experimentData.isRunning)

                ExperimentButton(
                    title: "View Real\nCharts",
                    icon: "chart.bar.fill",
                    color: .cyan,
                    description: "Live experimental data visualizations",
                    badge: nil,
                    action: showCharts
                )
            }

            HStack(spacing: 12) {
                Button(action: runAll) {
                    HStack {
                        if experimentData.isRunning { ProgressView().scaleEffect(0.8); Text("Running Real Experiments…") }
                        else { Image(systemName: "play.fill"); Text("Run Complete Real Experiment Suite") }
                    }
                    .padding(12)
                    .frame(maxWidth: .infinity)
                    .background(experimentData.isRunning ? Color.gray : Color.red)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                }
                .disabled(experimentData.isRunning)

                Menu {
                    Button("Export Real Data (JSON)", action: exportJSON)
                    Button("Export Real Charts (PNG)", action: exportCharts)
                    Button("Export Analysis Report", action: exportReport)
                    Divider()
                    Button("Export All Real Data", action: exportAll)
                } label: {
                    Label("Export Real Data", systemImage: "square.and.arrow.up")
                        .padding(12)
                        .background(Color.gray.opacity(0.1))
                        .foregroundColor(.primary)
                        .cornerRadius(10)
                }
                .disabled(experimentData.accuracyResults.isEmpty && experimentData.performanceResults.isEmpty)
            }
        }
        .padding()
        .background(Color.gray.opacity(0.03))
        .cornerRadius(12)
    }
}

// MARK: - Results Summary

private struct ResultsSummaryPanel: View {
    @ObservedObject var experimentData: ExperimentData

    var body: some View {
        Group {
            if !experimentData.accuracyResults.isEmpty || !experimentData.performanceResults.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Real Experiment Results Summary")
                            .font(.headline).fontWeight(.semibold)
                        Spacer()
                        let total = experimentData.accuracyResults.count
                                + experimentData.performanceResults.count
                                + experimentData.scalabilityResults.count
                        Text("\(total) real test cases")
                            .font(.caption).foregroundColor(.secondary)
                    }

                    ScrollView {
                        VStack(alignment: .leading, spacing: 8) {
                            if !experimentData.accuracyResults.isEmpty {
                                Text("Accuracy Results (\(experimentData.accuracyResults.count) tests):")
                                    .font(.subheadline).fontWeight(.semibold)
                                ForEach(experimentData.aggregatedAccuracyData, id: \.0) { engine, acc, _ in
                                    Text("• \(engine): \(String(format: "%.1f%%", acc)) accuracy").font(.caption)
                                }
                            }

                            if !experimentData.performanceResults.isEmpty {
                                Text("\nPerformance Results (\(experimentData.performanceResults.count) methods):")
                                    .font(.subheadline).fontWeight(.semibold)
                                ForEach(experimentData.performanceResults, id: \.method.rawValue) { r in
                                    Text("• \(r.method.rawValue): \(String(format: "%.1f", r.averageLatencyMs))ms avg")
                                        .font(.caption)
                                }
                            }

                            if !experimentData.performanceInsights.isEmpty {
                                Text("\nKey Insights from Real Data:")
                                    .font(.subheadline).fontWeight(.semibold)
                                ForEach(experimentData.performanceInsights.prefix(3), id: \.self) { insight in
                                    Text(insight).font(.caption)
                                }
                            }
                        }
                    }
                    .frame(height: 150)
                    .background(Color.blue.opacity(0.05))
                    .cornerRadius(8)
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.blue.opacity(0.3), lineWidth: 1))
                }
            }
        }
    }
}

// MARK: - Updated Chart Views with Better Visibility

