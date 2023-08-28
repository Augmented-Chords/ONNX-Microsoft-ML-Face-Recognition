using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Security.RightsManagement;
using System.Text;
using System.Threading.Tasks;


namespace FaceRecognition
{
    public class PredictionResult
    {
        public Bitmap bitmap;
        public Rectangle rectangle;
        public string emotion;
        public string age;
        public string gender;

        public PredictionResult()
        {
            bitmap = new Bitmap(0, 0);
            rectangle = new Rectangle();
            emotion = "";
            age = "";
            gender = "";
        }

        public PredictionResult(Bitmap givenBitmap, Rectangle givenRectangle)
        {
            bitmap = givenBitmap;
            rectangle = givenRectangle;
            emotion = "";
            age = "";
            gender = "";
        }
    }

    public class FaceRecognition : IDisposable
    {
        private readonly string modelFile = "onnx-models/version-RFB-320.onnx";
        private PredictionEngine<FaceRecognitionInput, FaceRecognitionOutput> predictionEngine = null;
        public FaceRecognition()
        {
            MLContext context = new MLContext();
            var emptyData = new List<FaceRecognitionInput>();
            var data = context.Data.LoadFromEnumerable(emptyData);
            var pipeline = context.Transforms.ResizeImages("resize", 320, 240, nameof(FaceRecognitionInput.Image), ImageResizingEstimator.ResizingKind.Fill).
                Append(context.Transforms.ExtractPixels("input", "resize", offsetImage: 127f, scaleImage: 1 / 128f)).
                Append(context.Transforms.ApplyOnnxModel(modelFile: modelFile, inputColumnNames: new string[] { "input" }, outputColumnNames: new string[] { "scores", "boxes" }));
            var model = pipeline.Fit(data);
            predictionEngine = context.Model.CreatePredictionEngine<FaceRecognitionInput, FaceRecognitionOutput>(model);
        }

        public void Dispose()
        {
            predictionEngine.Dispose();

        }

        public PredictionResult[] Predict(Bitmap bitmap)
        {
            var result = predictionEngine.Predict(new FaceRecognitionInput() { Image = BitmapUtility.ConvertBitmapToMLImage(bitmap) });
            var length = result.Boxes.Length / 4;
            var faceRecognitionEnumerableResult = Enumerable.Range(0, length).Select(i => new FaceRecognitionResult()
            {
                rectangle = new Rectangle((int)(result.Boxes[i * 4] * bitmap.Width), (int)(result.Boxes[i * 4 + 1] * bitmap.Height), (int)((result.Boxes[i * 4 + 2] - result.Boxes[i * 4]) * bitmap.Width), (int)((result.Boxes[i * 4 + 3] - result.Boxes[i * 4 + 1]) * bitmap.Height)),
                score = result.Scores[i * 2 + 1]
            });
            faceRecognitionEnumerableResult = faceRecognitionEnumerableResult.Where(faceRecognitionEnumerableResult => faceRecognitionEnumerableResult.rectangle.Height > 0 && faceRecognitionEnumerableResult.rectangle.Bottom < bitmap.Height && faceRecognitionEnumerableResult.rectangle.Width > 0 && faceRecognitionEnumerableResult.rectangle.Right < bitmap.Width && faceRecognitionEnumerableResult.score > 0.8);
            var faceRecognitionListResult = faceRecognitionEnumerableResult.ToList();
            faceRecognitionListResult = NonMaximumSuppression(faceRecognitionListResult, 0.4);
            var bitmaps = new Bitmap[faceRecognitionListResult.Count];
            var predictionResults = new PredictionResult[faceRecognitionListResult.Count];
            for (int i = 0; i < faceRecognitionListResult.Count; i++)
            {
                bitmaps[i] = bitmap.Clone(faceRecognitionListResult[i].rectangle, bitmap.PixelFormat);
                predictionResults[i] = new PredictionResult(bitmaps[i], faceRecognitionListResult[i].rectangle);
            }
            return predictionResults;
        }


        public List<FaceRecognitionResult> NonMaximumSuppression(List<FaceRecognitionResult> FaceRecognitionResults, double overlapThreshold)
        {
            //Order by score
            FaceRecognitionResults.OrderByDescending(FaceRecognitionResults => FaceRecognitionResults.score);
            var currentResults = new List<FaceRecognitionResult>();
            while (FaceRecognitionResults.Count > 0)
            {
                var currentResult = FaceRecognitionResults[0];
                currentResults.Add(currentResult);

                FaceRecognitionResults.RemoveAt(0);
                for (int i = FaceRecognitionResults.Count - 1; i >= 0; i--)
                {
                    var iou = IntersectionOverUnion(currentResult, FaceRecognitionResults[i]);
                    if (iou >= overlapThreshold)
                    {
                        FaceRecognitionResults.RemoveAt(i);
                    }
                }
            }
            //Order from left to right
            currentResults.OrderByDescending(currentResults => currentResults.rectangle.X);
            return currentResults;
        }

        public double IntersectionOverUnion(FaceRecognitionResult resultA, FaceRecognitionResult resultB)
        {
            float xOverlap = Math.Max(0, Math.Min(resultA.rectangle.Right, resultB.rectangle.Right) - Math.Max(resultA.rectangle.X, resultB.rectangle.X) + 1);
            float yOverlap = Math.Max(0, Math.Min(resultA.rectangle.Bottom, resultB.rectangle.Bottom) - Math.Max(resultA.rectangle.Y, resultB.rectangle.Y) + 1);
            float intersectionArea = xOverlap * yOverlap;
            float unionArea = resultA.rectangle.Width * resultA.rectangle.Height + resultB.rectangle.Width * resultB.rectangle.Height - intersectionArea;
            double iou = (double)intersectionArea / unionArea;
            return iou;
        }
    }

    public class FaceRecognitionInput
    {
        [ImageType(320, 240)]
        public MLImage Image { get; set; }
    }

    public class FaceRecognitionOutput
    {
        [ColumnName("scores")]
        [VectorType(1, 4420, 2)]
        public float[] Scores { get; set; }
        [ColumnName("boxes")]
        [VectorType(1, 4420, 4)]
        public float[] Boxes { get; set; }
    }

    public class FaceRecognitionResult
    {
        public float score;
        public Rectangle rectangle;
    }

    public class EmotionEstimator : IDisposable
    {
        private readonly string modelFile = "onnx-models/emotion-ferplus-8.onnx";
        private string[] emotions = new string[] { "Neutral", "Happiness", "Surprise", "Sadness", "Anger", "Disgust", "Fear", "Contempt" };
        private PredictionEngine<EmotionEstimatorInput, EmotionEstimatorOutput> predictionEngine;
        public EmotionEstimator()
        {
            MLContext context = new MLContext();
            var emptyData = new List<EmotionEstimatorInput>();
            var data = context.Data.LoadFromEnumerable(emptyData);
            var pipeline = context.Transforms.ResizeImages("resize", 64, 64, inputColumnName: nameof(EmotionEstimatorInput.Image), ImageResizingEstimator.ResizingKind.Fill).
                Append(context.Transforms.ConvertToGrayscale("gray", "resize")).
                Append(context.Transforms.ExtractPixels("Input3", "gray", ImagePixelExtractingEstimator.ColorBits.Blue)).
                Append(context.Transforms.ApplyOnnxModel(modelFile));

            var model = pipeline.Fit(data);
            predictionEngine = context.Model.CreatePredictionEngine<EmotionEstimatorInput, EmotionEstimatorOutput>(model);
        }

        public void Dispose()
        {
            predictionEngine.Dispose();
        }


        public void Predict(PredictionResult predictionResult)
        {
            var result = predictionEngine.Predict(new EmotionEstimatorInput() { Image = BitmapUtility.ConvertBitmapToMLImage(predictionResult.bitmap) });
            var max = result.Result.Max();
            var index = result.Result.ToList().IndexOf(max);
            predictionResult.emotion = emotions[index];
        }

        public void Predict(PredictionResult[] predictionResults)
        {
            foreach (PredictionResult predictionResult in predictionResults)
            {
                var result = predictionEngine.Predict(new EmotionEstimatorInput() { Image = BitmapUtility.ConvertBitmapToMLImage(predictionResult.bitmap) });
                var max = result.Result.Max();
                var index = result.Result.ToList().IndexOf(max);
                predictionResult.emotion = emotions[index];
            }
        }
    }
    public class EmotionEstimatorInput
    {
        [ImageType(64, 64)]
        public MLImage Image { get; set; }
    }

    public class EmotionEstimatorOutput
    {
        [ColumnName("Plus692_Output_0")]
        public float[] Result { get; set; }
    }

    public class AgeGenderEstimator : IDisposable
    {
        private readonly string ageModelFile = "onnx-models/age_googlenet.onnx";
        private readonly string genderModelFile = "onnx-models/gender_googlenet.onnx";
        private PredictionEngine<AgeGenderEstimatorInput, AgeGenderEstimatorOutput> agePredictionEngine;
        private PredictionEngine<AgeGenderEstimatorInput, AgeGenderEstimatorOutput> genderPredictionEngine;
        private string[] ageArray = new string[] { "0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100" };
        private string[] genderArray = new string[] { "Male", "Female" };
        public AgeGenderEstimator()
        {
            MLContext ageContext = new MLContext();
            var ageEmptyData = new List<AgeGenderEstimatorInput>();
            var ageData = ageContext.Data.LoadFromEnumerable(ageEmptyData);
            var agePipeline = ageContext.Transforms.ResizeImages("resize", 224, 224, nameof(AgeGenderEstimatorInput.Image), ImageResizingEstimator.ResizingKind.Fill).
                Append(ageContext.Transforms.ExtractPixels("input", "resize", offsetImage: 127f, scaleImage: 1 / 128f, orderOfExtraction: ImagePixelExtractingEstimator.ColorsOrder.ABGR)).
                Append(ageContext.Transforms.ApplyOnnxModel(ageModelFile));
            var ageModel = agePipeline.Fit(ageData);
            agePredictionEngine = ageContext.Model.CreatePredictionEngine<AgeGenderEstimatorInput, AgeGenderEstimatorOutput>(ageModel);

            MLContext genderContext = new MLContext();
            var genderEmptyData = new List<AgeGenderEstimatorInput>();
            var genderData = ageContext.Data.LoadFromEnumerable(genderEmptyData);
            var pipeline = genderContext.Transforms.ResizeImages("resize", 224, 224, nameof(AgeGenderEstimatorInput.Image), ImageResizingEstimator.ResizingKind.Fill).
                Append(genderContext.Transforms.ExtractPixels("input", "resize", offsetImage: 127f, scaleImage: 1 / 128f, orderOfExtraction: ImagePixelExtractingEstimator.ColorsOrder.ABGR)).
                Append(genderContext.Transforms.ApplyOnnxModel(genderModelFile));
            var genderModel = pipeline.Fit(genderData);
            genderPredictionEngine = genderContext.Model.CreatePredictionEngine<AgeGenderEstimatorInput, AgeGenderEstimatorOutput>(genderModel);
        }

        public void Predict(PredictionResult predictionResult)
        {
            var ageResult = agePredictionEngine.Predict(new AgeGenderEstimatorInput() { Image = BitmapUtility.ConvertBitmapToMLImage(predictionResult.bitmap) });
            var ageMax = ageResult.Result.Max();
            var ageIndex = ageResult.Result.ToList().IndexOf(ageMax);
            predictionResult.age = ageArray[ageIndex];
            var genderResult = genderPredictionEngine.Predict(new AgeGenderEstimatorInput() { Image = BitmapUtility.ConvertBitmapToMLImage(predictionResult.bitmap) });
            var genderMax = genderResult.Result.Max();
            var genderIndex = genderResult.Result.ToList().IndexOf(genderMax);
            predictionResult.gender = genderArray[genderIndex];
        }

        public void Predict(PredictionResult[] predictionResults)
        {
            foreach (PredictionResult predictionResult in predictionResults)
            {
                var ageResult = agePredictionEngine.Predict(new AgeGenderEstimatorInput() { Image = BitmapUtility.ConvertBitmapToMLImage(predictionResult.bitmap) });
                var ageMax = ageResult.Result.Max();
                var ageIndex = ageResult.Result.ToList().IndexOf(ageMax);
                predictionResult.age = ageArray[ageIndex];
                var genderResult = genderPredictionEngine.Predict(new AgeGenderEstimatorInput() { Image = BitmapUtility.ConvertBitmapToMLImage(predictionResult.bitmap) });
                var genderMax = genderResult.Result.Max();
                var genderIndex = genderResult.Result.ToList().IndexOf(genderMax);
                predictionResult.gender = genderArray[genderIndex];
            }
        }

        public void Dispose()
        {
            agePredictionEngine.Dispose();
            genderPredictionEngine.Dispose();
        }
    }


    public class AgeGenderEstimatorInput
    {
        [ImageType(224, 224)]
        public MLImage Image { get; set; }
    }

    public class AgeGenderEstimatorOutput
    {
        [ColumnName("loss3/loss3_Y")]
        public float[] Result { get; set; }
    }
}

