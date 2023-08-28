using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using System.Windows.Media;
using System.Drawing;
using System.Drawing.Imaging;
using Microsoft.ML.Data;
using System.Runtime.InteropServices;
using System.IO;
using System.Windows.Media.Media3D;

namespace FaceRecognition
{
    class BitmapUtility
    {
        public static Bitmap Resize(Bitmap bitmap, int targetHeight)
        {
            int targetWidth = bitmap.Width * targetHeight / bitmap.Height;
            return Resize(bitmap, targetWidth, targetHeight);
        }

        /// <summary>
        /// Resize Bitmap to given Max Width and Max Height.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="maxWidth"></param>
        /// <param name="maxHeight"></param>
        /// <returns></returns>
        public static Bitmap Resize(Bitmap input, int maxWidth, int maxHeight)
        {
            try
            {
                var ratioX = (double)maxWidth / input.Width;
                var ratioY = (double)maxHeight / input.Height;
                var ratio = Math.Min(ratioX, ratioY);

                var newWidth = (int)(input.Width * ratio);
                var newHeight = (int)(input.Height * ratio);

                var actualBitmap = new Bitmap(newWidth, newHeight);

                var g = Graphics.FromImage(actualBitmap);
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.Default; //Set InterpolationMode

                g.DrawImage(input, 
                    new Rectangle(0, 0, newWidth, newHeight),
                    new Rectangle(0, 0, input.Width, input.Height), 
                    GraphicsUnit.Pixel);
                g.Dispose();
                return actualBitmap;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Bitmap resize error. {ex.Message}");
                return null;
            }
        }

        public static BitmapSource ConvertBitmapToBitmapSource(System.Drawing.Bitmap bitmap)
        {
            var bitmapData = bitmap.LockBits(
                new Rectangle(0, 0, bitmap.Width, bitmap.Height),
                ImageLockMode.ReadOnly,
                System.Drawing.Imaging.PixelFormat.Format24bppRgb);

            var bitmapSource = BitmapSource.Create(
                bitmapData.Width, bitmapData.Height,
                bitmap.HorizontalResolution, bitmap.VerticalResolution,
                PixelFormats.Bgr24, null,
                bitmapData.Scan0, bitmapData.Stride * bitmapData.Height, bitmapData.Stride);

            bitmap.UnlockBits(bitmapData);

            return bitmapSource;
        }
        public static MLImage ConvertBitmapToMLImage(Bitmap bitmap)
        {
            MemoryStream ms = new();
            bitmap.Save(ms, ImageFormat.Bmp);
            ms.Seek(0, SeekOrigin.Begin);
            MLImage mLImage = MLImage.CreateFromStream(ms);
            ms.Dispose();
            return mLImage;
        }
    }
}
