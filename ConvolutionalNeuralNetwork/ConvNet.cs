using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNeuralNetwork
{
    public class ConvNet
    {
        public ConvNet()
        {

        }

        public double[,,] Convolutional(double[,,] input, double[,,,] filter)
        {
            double[,,] output = new double[input.GetLength(0), input.GetLength(1), input.GetLength(2)];

            for (int i = 0; i < filter.GetLength(0); i++)
            {
                for (int j = 0; j < input.GetLength(0); j++)
                {
                    for (int k = 0; k < input.GetLength(1); k++)
                    {
                        for (int l = 0; l < input.GetLength(2); l++)
                        {
                            output[j, k, l] = input[j, k, l] * filter[i, j, j, j];
                        }
                    }
                }
            }

            return output;
        }

        public double[,,] MaxPooling(double[,,] input, int filtersize)
        {
            double[,,] output = null;
            var newHeight = ((input.GetLength(1) - filtersize) / 2) + 1;
            var newWidth = ((input.GetLength(2) - filtersize) / 2) + 1;

            output = new double[input.GetLength(0), newHeight, newWidth];

            for (int j = 0; j <= 2; j++)
            {
                var cuurentY = 0;
                var outY = 0;

                for (int k = 0; k <= 15; k++)
                {
                    var cuurentX = 0;
                    var outX = 0;

                    for (int l = 0; l <= 15; l++)
                    {
                        double maxValue = MaxValue(input, j, k, l, filtersize);
                        output[j, outY, outX] = input[j, k, l] > maxValue ? input[j, k, l] : maxValue;
                        cuurentX = cuurentX + 2;
                        outX = outX + 1;
                    }

                    cuurentY = cuurentY + 2;
                    outY = outY + 1;
                }
            }           
            
            return output;
        }

        private double MaxValue(double[,,] input, int j, int k, int l, int filtersize)
        {
            double maxValue = 0;
            
            for (int a = 0; a < k + filtersize; a++)
            {
                for (int b = 0; b < l + filtersize; b++)
                {
                    maxValue = maxValue < input[j, a, b] ? input[j, a, b] : maxValue;
                }
            }

            return maxValue;
        }

        public double[] Flatten(double[,,] input)
        {
            int rgbChannel = input.GetLength(0);
            int rowPixel = input.GetLength(1);
            int columnPixel = input.GetLength(2);
            int length = rgbChannel * rowPixel * columnPixel;
            double[] output = new double[length];

            int count = 0;
            for (int i = 0; i < rgbChannel; i++)
            {
                for (int j = 0; j < rowPixel; j++)
                {
                    for (int k = 0; k < columnPixel; k++)
                    {
                        output[count] = input[i, j, k];
                        count = count + 1;
                    }
                }
            }

            return output;
        }

        public double FullyConnected(double[] input, double[] weights)
        {
            double sum = 0;

            for (int i = 0; i < input.Length; i++)
            {
                sum = sum + (input[i] * weights[i]);
            }

            return sum;
        }

        public double[,,] RectifiedLinearUnit(double[,,] input)
        {
            double[,,] output = new double[input.GetLength(0), input.GetLength(1), input.GetLength(2)];

            for (int j = 0; j < input.GetLength(0); j++)
            {
                for (int k = 0; k < input.GetLength(1); k++)
                {
                    for (int l = 0; l < input.GetLength(2); l++)
                    {
                        output[j, k, l] = input[j, k, l] < 0 ? 0 : input[j, k, l];
                    }
                }
            }

            return output;
        }

        public double[,,,] Filter(int filter, int nooffilters, int pixelsize)
        {
            double[,,,] doubleFilter = new double[filter, nooffilters, pixelsize, pixelsize];
            Random random = new Random();
            for (int i = 0; i < filter; i++)
            {
                for (int j = 0; j < nooffilters; j++)
                {
                    for (int k = 0; k < pixelsize; k++)
                    {
                        for (int l = 0; l < pixelsize; l++)
                        {
                            doubleFilter[i, j, k, l] = random.NextDouble();
                        }
                    }
                }
            }

            return doubleFilter;
        }

        public double[] RandomWeights(int count)
        {
            double[] weights = new double[count];
            Random random = new Random();

            for (int i = 0; i < count; i++)
            {
                weights[i] = random.NextDouble();
            }

            return weights;
        }

        public void TrainingForConvolutionalNeuralNetwork()
        {
            Bitmap img = new Bitmap(@"C:\temp\Test.JPG", true);
            double[,,] pixelvalues = new double[3, img.Width, img.Height];

            for (int i = 0; i < img.Width; i++)
            {
                for (int j = 0; j < img.Height; j++)
                {
                    Color pixel = img.GetPixel(i, j);
                    pixelvalues[0, i, j] = pixel.R;
                    pixelvalues[1, i, j] = pixel.G;
                    pixelvalues[2, i, j] = pixel.B;                  
                }
            }            

            var filters = this.Filter(1, 3, 3);
            var convolutionOutput = Convolutional(pixelvalues, filters);
            var activationOutput = RectifiedLinearUnit(convolutionOutput);
            
            var maxPoolingOutput = MaxPooling(activationOutput, 2);            
            var flatternOutput = Flatten(maxPoolingOutput);
            double[] weights = this.RandomWeights(flatternOutput.Length);
            
            var fullyConnectedOutput = FullyConnected(flatternOutput, weights);           

            //TODO
            //Evaluation
        }
    }
}
