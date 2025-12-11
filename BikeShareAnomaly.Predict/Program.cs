using Microsoft.ML;
using System;
using System.IO;
using System.Linq;

namespace BikeShareAnomaly.Predict
{
    public static class Predictor
    {
        public static void Predict(string csvPath)
        {
            var ml = new MLContext();

            if (!File.Exists(csvPath))
            {
                Console.WriteLine("❌ CSV not found: " + csvPath);
                return;
            }

            // 1) Завантажуємо CSV
            var data = ml.Data.LoadFromTextFile<ModelInput>(
                csvPath, separatorChar: ',', hasHeader: true);

            // 2) Створюємо ТАКИЙ ЖЕ pipeline
            var pipeline = ml.Transforms.DetectAnomalyBySrCnn(
                outputColumnName: "Prediction",
                inputColumnName: "cnt",
                windowSize: 32
            );

            // 3) Fit() — знову навчаємо трансформер
            var model = pipeline.Fit(data);

            // 4) Transform() — прогноз
            var transformed = model.Transform(data);

            var results = ml.Data.CreateEnumerable<ModelOutput>(transformed, reuseRowObject: false).ToList();

            Console.WriteLine("=== PREDICTION RESULTS (first 20) ===");

            int index = 0;
            foreach (var r in results.Take(20))
            {
                Console.WriteLine($"{index}: anomaly={r.Prediction[0]} score={r.Prediction[1]:F4} p={r.Prediction[2]:F4}");
                index++;
            }

            Console.WriteLine("🎯 Prediction completed");
        }
    }
}
