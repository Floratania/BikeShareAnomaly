using Microsoft.ML.Data;

namespace BikeShareAnomaly
{
    public class ModelInput
    {
        [LoadColumn(1)]
        public string dteday { get; set; }

        [LoadColumn(5)]
        public float hr { get; set; }

        [LoadColumn(16)]
        public float cnt { get; set; }
    }
}
