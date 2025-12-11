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

            Console.WriteLine("TRAIN: SR-CNN ANOMALY DETECTOR ");
            Console.WriteLine("CSV PATH: " + csvPath);

            if (!File.Exists(csvPath))
            {
                Console.WriteLine(" CSV not found!");
                return;
            }

  
            var rawData = ml.Data.LoadFromTextFile<ModelInput>(
                csvPath, hasHeader: true, separatorChar: ',');

            // 2) Convert to list and compute Timestamp
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

            // 3) Convert back to IDataView *after sorting*
            var data = ml.Data.LoadFromEnumerable(list);

            Console.WriteLine($"📌 Total rows: {list.Count}");


            var pipeline = ml.Transforms.DetectAnomalyBySrCnn(
                outputColumnName: "Prediction",
                inputColumnName: "cnt",
                windowSize: 48 
            );

            var model = pipeline.Fit(data);
            var transformed = model.Transform(data);

            var results = ml.Data.CreateEnumerable<ModelOutput>(transformed, false).ToList();

            Console.WriteLine("\n1500 - 1550 rows");
            for (int i = 1500; i < 1550; i++)
            {
                Console.WriteLine($"{i}: anomaly={results[i].Prediction[0]}  score={results[i].Prediction[1]:F4}  p={results[i].Prediction[2]:F4}");
            }

           

            Console.WriteLine("\nTRAIN completed (model is NOT saved — ML.NET 5.0 limitation)");
        }
    }
}
