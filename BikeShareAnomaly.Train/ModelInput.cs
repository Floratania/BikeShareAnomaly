using Microsoft.ML.Data;

namespace BikeShareAnomaly
{
    public class ModelInput
    {
        // Колонка cnt — 17-та в CSV (індекс 16)
        [LoadColumn(16)]
        public float cnt { get; set; }
    }
}
