using Microsoft.ML.Data;

namespace BikeShareAnomaly
{
    public class ModelOutput
    {
        // SR-CNN always returns 3 values:
        // [0] - isAnomaly (0/1)
        // [1] - score
        // [2] - p-value
        [VectorType(3)]
        public double[] Prediction { get; set; }
    }
}
