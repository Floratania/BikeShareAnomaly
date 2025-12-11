using System;
using System.IO;
using BikeShareAnomaly.Train;
using BikeShareAnomaly.Predict;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("1 — Train");
        Console.WriteLine("2 — Predict");
        Console.Write("Choose: ");

        char key = Console.ReadKey().KeyChar;
        Console.WriteLine();

        string root = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, @"..\..\.."));
        string csvPath = Path.Combine(root, "Data", "bike_sharing.csv");

        if (key == '1')
            Trainer.Train();
        else if (key == '2')
            Predictor.Predict(csvPath);
        else
            Console.WriteLine("Unknown option.");
    }
}
