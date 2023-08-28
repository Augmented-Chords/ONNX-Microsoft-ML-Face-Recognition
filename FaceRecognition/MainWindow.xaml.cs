using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Media.Media3D;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace FaceRecognition
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            // Initiate OpenFileDialog
            Microsoft.Win32.OpenFileDialog dialog = new Microsoft.Win32.OpenFileDialog();
            dialog.Filter = "Image file|*.bmp;*.png;*.jpg;*.jpeg";
            // Show the diaglog
            Nullable<bool> result = dialog.ShowDialog();
            if (result == true)
            {
                string file = dialog.FileName; // Get selected file name
                Bitmap bitmap = new Bitmap(file);
                PredictionResult[] predictionResults;
                using FaceRecognition faceRecognition = new FaceRecognition();
                using EmotionEstimator emotionEstimator = new EmotionEstimator();
                using AgeGenderEstimator ageGenderEstimator = new AgeGenderEstimator();
                predictionResults = faceRecognition.Predict(bitmap);
                emotionEstimator.Predict(predictionResults);
                ageGenderEstimator.Predict(predictionResults);
                using (Graphics graphics = Graphics.FromImage(bitmap))
                {
                    // Create a red pen
                    System.Drawing.Pen redPen = new System.Drawing.Pen(System.Drawing.Color.Red, 1);
                    Font drawFont = new Font("Arial", 16);
                    SolidBrush drawBrush = new SolidBrush(System.Drawing.Color.Red);
                    StringFormat drawFormat = new StringFormat();
                    drawFormat.Alignment = StringAlignment.Near;
                    // Loop through the rectangles and draw them on the bitmap
                    foreach (var predictionResult in predictionResults)
                    {
                        graphics.DrawRectangle(redPen, predictionResult.rectangle);
                        RectangleF drawRect = new RectangleF(predictionResult.rectangle.X, predictionResult.rectangle.Bottom, 512, 512);
                        graphics.DrawString(predictionResult.emotion + "\n" + predictionResult.age + "\n" + predictionResult.gender, drawFont, drawBrush, drawRect, drawFormat);
                    }
                }
                Image.Source = BitmapUtility.ConvertBitmapToBitmapSource(BitmapUtility.Resize(bitmap, 580, 494));

            }
        }
    }
}
