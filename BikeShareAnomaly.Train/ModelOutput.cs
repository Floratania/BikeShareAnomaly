using Microsoft.ML.Data;

namespace BikeShareAnomaly
{
    public class ModelOutput
    {
        // SR-CNN повертає 3 значення
        [VectorType(3)]
        public double[] Prediction { get; set; }
    }
}
