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
                Console.WriteLine(" CSV not found: " + csvPath);
                return;
            }

            var rawData = ml.Data.LoadFromTextFile<ModelInput>(
                csvPath, separatorChar: ',', hasHeader: true);

            var list = ml.Data.CreateEnumerable<ModelInput>(rawData, false)
                .Select(x => new
                {
                    x.dteday,
                    x.hr,
                    x.cnt,
                    Timestamp = DateTime.Parse(x.dteday).AddHours(x.hr)
                })
                .OrderBy(x => x.Timestamp)
                .ToList();

  
            var data = ml.Data.LoadFromEnumerable(list);

            var pipeline = ml.Transforms.DetectAnomalyBySrCnn(
                outputColumnName: "Prediction",
                inputColumnName: "cnt",
                windowSize: 48
            );

            var model = pipeline.Fit(data);
            var transformed = model.Transform(data);

            var results = ml.Data.CreateEnumerable<ModelOutput>(transformed, false).ToList();

            Console.WriteLine(" PREDICTION RESULTS (2100 - 2200)");

            for (int i = 2100; i < 2200; i++)
            {
                Console.WriteLine($"{i}: anomaly={results[i].Prediction[0]}  score={results[i].Prediction[1]:F4}  p={results[i].Prediction[2]:F4}");
            }

            Console.WriteLine(" Prediction completed");
        }
    }
}
