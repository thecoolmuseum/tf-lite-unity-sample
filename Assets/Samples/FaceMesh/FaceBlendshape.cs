using System.Collections;
using System.Collections.Generic;

using System.IO;
using System;

using UnityEngine;


namespace TensorFlowLite
{
    public class FaceBlendshape : System.IDisposable
    {
        public class Result
        {
            public float[] blendshapes;
        }
        public const int BLENDSHAPE_COUNT = 52;
        private Interpreter interpreter;
        protected float[,] input0; // facepoints
        protected float[] output0 = new float[BLENDSHAPE_COUNT]; // blendshapes
        private Result result;

        protected static int[] facePoints = new int[]{
            0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80,
            81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157,
            158, 159, 160, 161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246, 249, 251, 263, 267, 269, 270, 276, 282,
            283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334, 336, 338, 356,
            361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405,
            409, 415, 454, 466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477
        };
        public static string[] BlendshapeNames = new string[] {
            "_neutral",
            "browDownLeft",
            "browDownRight",
            "browInnerUp",
            "browOuterUpLeft",
            "browOuterUpRight",
            "cheekPuff",
            "cheekSquintLeft",
            "cheekSquintRight",
            "eyeBlinkLeft",
            "eyeBlinkRight",
            "eyeLookDownLeft",
            "eyeLookDownRight",
            "eyeLookInLeft",
            "eyeLookInRight",
            "eyeLookOutLeft",
            "eyeLookOutRight",
            "eyeLookUpLeft",
            "eyeLookUpRight",
            "eyeSquintLeft",
            "eyeSquintRight",
            "eyeWideLeft",
            "eyeWideRight",
            "jawForward",
            "jawLeft",
            "jawOpen",
            "jawRight",
            "mouthClose",
            "mouthDimpleLeft",
            "mouthDimpleRight",
            "mouthFrownLeft",
            "mouthFrownRight",
            "mouthFunnel",
            "mouthLeft",
            "mouthLowerDownLeft",
            "mouthLowerDownRight",
            "mouthPressLeft",
            "mouthPressRight",
            "mouthPucker",
            "mouthRight",
            "mouthRollLower",
            "mouthRollUpper",
            "mouthShrugLower",
            "mouthShrugUpper",
            "mouthSmileLeft",
            "mouthSmileRight",
            "mouthStretchLeft",
            "mouthStretchRight",
            "mouthUpperUpLeft",
            "mouthUpperUpRight",
            "noseSneerLeft",
            "noseSneerRight"
        };


        public FaceBlendshape(string modelPath, bool useGPU = true)
        {
            result = new Result()
            {
                blendshapes = new float[BLENDSHAPE_COUNT],
            };

            var options = new InterpreterOptions();
            if (useGPU)
            {
                options.AddGpuDelegate();
            }
            else
            {
                options.threads = SystemInfo.processorCount;
            }

            // Blendshape用のInterpreter
            interpreter = new Interpreter(FileUtil.LoadFile(modelPath), options);
            interpreter.LogIOInfo();
            var idim0 = interpreter.GetInputTensorInfo(0).shape;
            var count = idim0[1];
            var channels = idim0[2];
            input0 = new float[count, channels];
            int inputCount = interpreter.GetInputTensorCount();
            for (int i = 0; i < inputCount; i++)
            {
                int[] dim = interpreter.GetInputTensorInfo(i).shape;
                interpreter.ResizeInputTensor(i, dim);
            }
            interpreter.AllocateTensors();
        }

        public virtual void Dispose()
        {
            interpreter?.Dispose();
       }

        public void Invoke(FaceMesh.Result face)
        {
            // Blendshape
            for (int i = 0; i < facePoints.Length; i++)
            {
                input0[i, 0] = face.keypoints[facePoints[i]][0] / 256f;
                input0[i, 1] = face.keypoints[facePoints[i]][1] / 256f;
            }
            interpreter.SetInputTensorData(0, input0);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, output0);
        }

        public Result GetResult()
        {
            for (int i = 0; i < BLENDSHAPE_COUNT; i++)
            {
                result.blendshapes[i] = output0[i];
            }
            return result;
        }

    }
}
