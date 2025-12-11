using Microsoft.ML;
using System;
using System.IO;
using System.Linq;

namespace BikeShareAnomaly.Train
{
    public static class Trainer
    {
        public static void Train()
        {
            var ml = new MLContext();
            Console.OutputEncoding = System.Text.Encoding.UTF8;

            string projectRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, @"..\..\.."));
            string csvPath = Path.Combine(projectRoot, "Data", "bike_sharing.csv");

            Console.WriteLine("=== TRAINING SR-CNN ANOMALY DETECTOR ===");
            Console.WriteLine("CSV PATH: " + csvPath);

            if (!File.Exists(csvPath))
            {
                Console.WriteLine("❌ CSV file not found!");
                return;
            }

            // 1) Завантажуємо дані
            var data = ml.Data.LoadFromTextFile<ModelInput>(
                csvPath, hasHeader: true, separatorChar: ',');

            var rowCount = ml.Data.CreateEnumerable<ModelInput>(data, reuseRowObject: true).Count();
            Console.WriteLine($"📌 Rows loaded: {rowCount}");

            // 2) Створюємо pipeline SR-CNN
            var pipeline = ml.Transforms.DetectAnomalyBySrCnn(
                outputColumnName: "Prediction",
                inputColumnName: "cnt",
                windowSize: 32
            );

            // 3) Fit() — SR-CNN виконує тренування внутрішньо
            var model = pipeline.Fit(data);

            // 4) Transform() — отримуємо аномалії
            var transformed = model.Transform(data);

            var results = ml.Data.CreateEnumerable<ModelOutput>(transformed, reuseRowObject: false).ToList();

            Console.WriteLine();
            Console.WriteLine("=== FIRST 20 ANOMALY RESULTS ===");

            int index = 0;
            foreach (var r in results.Take(20))
            {
                Console.WriteLine(
                    $"{index,4}: anomaly={r.Prediction[0]} score={r.Prediction[1]:F4} p={r.Prediction[2]:F4}"
                );
                index++;
            }

            Console.WriteLine();
            Console.WriteLine("🎯 TRAIN completed (model NOT saved — ML.NET 5.0 restriction)");
        }
    }
}
