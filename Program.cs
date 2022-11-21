using DlibDotNet;
using OpenCvSharp;
using System.Runtime.InteropServices;

namespace ApplyMasksForWorldCup
{
    internal static class Program
    {
        static void Main()
        {
            using var videoCapture = new VideoCapture("WorldCup2022_1.mp4", VideoCaptureAPIs.ANY);
            using var matMask = Cv2.ImRead("mask_small.png",ImreadModes.Unchanged);
            
            using var fd = Dlib.GetFrontalFaceDetector();
            using var sp = ShapePredictor.Deserialize("shape_predictor_68_face_landmarks.dat");
            
            using Mat matSrcFrame = new Mat();
            while(videoCapture.Read(matSrcFrame))
            {
                using var img = LoadDLibImageFromMat(matSrcFrame);
                var faces = fd.Operator(img);
                foreach (var face in faces)
                {
                    var shape = sp.Detect(img, face);
                    //https://www.studytonight.com/post/dlib-68-points-face-landmark-detection-with-opencv-and-python
                    var pointNoseTip = shape.GetPart(31);
                    var pointCheekLeft = shape.GetPart(3);
                    var pointCheekRight = shape.GetPart(13);
                    var pointChinBottom = shape.GetPart(8);

                    int maskWidth = pointCheekRight.X - pointCheekLeft.X;
                    int maskHeight = pointChinBottom.Y - pointNoseTip.Y;
                    using var matResizedMask = new Mat();
                    Cv2.Resize(matMask, matResizedMask,new (maskWidth, maskHeight));
                    if(maskWidth > 0&& maskHeight > 0)
                    {                        
                        DrawOverlay(matSrcFrame, matResizedMask, pointCheekLeft.X, pointNoseTip.Y);
                    }                    
                }

                Cv2.ImShow("0 Ti Qing 0",matSrcFrame);
                Cv2.WaitKey(1);
            }
        }

        public unsafe static void DrawOverlay(Mat bg, Mat overlay,int x,int y)
        {
            if (overlay.Channels() < 4)
            {
                throw new System.ArgumentException("overlay.Channels()<4");
            }
            int colsOverlay = overlay.Cols;
            int rowsOverlay = overlay.Rows;
            int bgHeight = bg.Height;
            int bgWidth = bg.Width;
            for (int i = 0; i < rowsOverlay; i++)
            {
                int yBg = y + i;
                if (yBg >= bgHeight) return;
                Vec3b* pBg = (Vec3b*)bg.Ptr(yBg);
                Vec4b* pOverlay = (Vec4b*)overlay.Ptr(i);
                for (int j = 0; j < colsOverlay; j++)
                {
                    int xBg = x + j;
                    if (xBg >= bgWidth) return;
                    Vec3b* pointBg = pBg + xBg;
                    Vec4b* pointOverlay = pOverlay + j;
                    if (pointOverlay->Item3 != 0)
                    {
                        pBg->Item0 = pOverlay->Item0;
                        pBg->Item1 = pOverlay->Item1;
                        pBg->Item2= pOverlay->Item2;
                    }
                }
            }
        }

        static Array2D<BgrPixel> LoadDLibImageFromMat(Mat mat)
        {
            int width = mat.Width;
            int height = mat.Height;
            int elemSize = mat.ElemSize();
            var array = new byte[width * height * elemSize];
            Marshal.Copy(mat.Data, array, 0, array.Length);
            var img = Dlib.LoadImageData<BgrPixel>(array, (uint)height, (uint)width, (uint)(width * elemSize));
            return img;
        }
    }
}