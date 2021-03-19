using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConvolutionalNeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            ConvNet convNet = new ConvNet();
            convNet.TrainingForConvolutionalNeuralNetwork();
        }
    }
}
