using System.Collections.Generic;
using System.Linq;
using System.Threading;
using UnityEngine;
using Cysharp.Threading.Tasks;

namespace TensorFlowLite
{

    public class PalmDetect : BaseImagePredictor<float>
    {
        public struct Result
        {
            public float score;
            public Rect rect;
            public Vector2[] keypoints;
        }

        public const int MAX_PALM_NUM = 4;

        // classificators / scores
        // private readonly float[] output0 = new float[2944];
        private readonly float[] output1 = new float[2016];

        // regressors / points
        // 0 - 3 are bounding box offset, width and height: dx, dy, w ,h
        // 4 - 17 are 7 hand keypoint x and y coordinates: x1,y1,x2,y2,...x7,y7
        // private readonly float[,] output1 = new float[2944, 18];
        private readonly float[,] output0 = new float[2016, 18];
        private readonly List<Result> results = new List<Result>();
        private readonly SsdAnchor[] anchors;

        public PalmDetect(string modelPath) : base(modelPath, Accelerator.GPU)
        {
            var options = new SsdAnchorsCalculator.Options()
            {
                inputSizeWidth = 192,
                inputSizeHeight = 192,

                minScale = 0.1484375f,
                maxScale = 0.75f,

                anchorOffsetX = 0.5f,
                anchorOffsetY = 0.5f,

                numLayers = 4,
                featureMapWidth = new int[0],
                featureMapHeight = new int[0],
                strides = new int[] { 8, 16, 16, 16 },

                aspectRatios = new float[] { 1.0f },

                reduceBoxesInLowestLayer = false,
                interpolatedScaleAspectRatio = 1.0f,
                fixedAnchorSize = true,
            };

            anchors = SsdAnchorsCalculator.Generate(options);
            Debug.Log(anchors.Length);
            Debug.AssertFormat(anchors.Length == 2016, "Anchors count must be 2016");

            // shape配列の内容を表示

            Debug.Log(string.Join(", ", interpreter.GetInputTensorInfo(0).shape));
            Debug.Log(string.Join(", ", interpreter.GetOutputTensorInfo(0).shape));
            Debug.Log(string.Join(", ", interpreter.GetOutputTensorInfo(1).shape));
        }

        public override void Invoke(Texture inputTex)
        {
            // const float OFFSET = 128f;
            // const float SCALE = 1f / 128f;
            // ToTensor(inputTex, input0, OFFSET, SCALE);
            ToTensor(inputTex, inputTensor);


            interpreter.SetInputTensorData(0, inputTensor);
            interpreter.Invoke();

            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);
        }

        public async UniTask<List<Result>> InvokeAsync(Texture inputTex, CancellationToken cancellationToken)
        {
            await ToTensorAsync(inputTex, inputTensor, cancellationToken);
            await UniTask.SwitchToThreadPool();

            interpreter.SetInputTensorData(0, inputTensor);
            interpreter.Invoke();

            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);

            var results = GetResults();

            await UniTask.SwitchToMainThread(cancellationToken);
            return results;
        }

        public List<Result> GetResults(float scoreThreshold = 0.7f, float iouThreshold = 0.3f)
        {
            results.Clear();

            for (int i = 0; i < anchors.Length; i++)
            {
                float score = MathTF.Sigmoid(output1[i]);
                if (score < scoreThreshold)
                {
                    continue;
                }

                SsdAnchor anchor = anchors[i];

                float sx = output0[i, 0];
                float sy = output0[i, 1];
                float w = output0[i, 2];
                float h = output0[i, 3];

                float cx = sx + anchor.x * width;
                float cy = sy + anchor.y * height;

                cx /= (float)width;
                cy /= (float)height;
                w /= (float)width;
                h /= (float)height;

                var keypoints = new Vector2[7];
                for (int j = 0; j < 7; j++)
                {
                    float lx = output0[i, 4 + (2 * j) + 0];
                    float ly = output0[i, 4 + (2 * j) + 1];
                    lx += anchor.x * width;
                    ly += anchor.y * height;
                    lx /= (float)width;
                    ly /= (float)height;
                    keypoints[j] = new Vector2(lx, ly);
                }

                results.Add(new Result()
                {
                    score = score,
                    rect = new Rect(cx - w * 0.5f, cy - h * 0.5f, w, h),
                    keypoints = keypoints,
                });

            }

            return NonMaxSuppression(results, iouThreshold);
        }

        private static List<Result> NonMaxSuppression(List<Result> palms, float iou_threshold)
        {
            var filtered = new List<Result>();

            foreach (Result originalPalm in palms.OrderByDescending(o => o.score))
            {
                bool ignore_candidate = false;
                foreach (Result newPalm in filtered)
                {
                    float iou = originalPalm.rect.IntersectionOverUnion(newPalm.rect);
                    if (iou >= iou_threshold)
                    {
                        ignore_candidate = true;
                        break;
                    }
                }

                if (!ignore_candidate)
                {
                    filtered.Add(originalPalm);
                    if (filtered.Count >= MAX_PALM_NUM)
                    {
                        break;
                    }
                }
            }

            return filtered;
        }

    }
}
