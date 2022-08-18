import argparse
import os

def getParsedArguments():
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-b", "--basePath", help = "basePath")
    parser.add_argument("-d", "--dataPath", help = 'path to data directory, default: {basePath}/data')
    parser.add_argument("-o", "--output", help = "Output directory, default: {basePath}/output")
    parser.add_argument("-s", "--sampleSize", help = "Samplesize")
    parser.add_argument("-f", "--filename", help = "Filename, optional")
    parser.add_argument("-m", "--modelVersion", help = "modelVersion, default: aparent")
    parser.add_argument("-dv", "--dataVersion", help = "dataVersion, default: 1")

    # Read arguments from command line
    args = parser.parse_args()

    if not args.basePath:
        raise Exception('Es wurde kein basePath angegeben.')

    basePath = os.path.join('', args.basePath)

    dataPath = os.path.join(basePath, 'data')
    if args.dataPath:
        dataPath = os.path.join('', args.dataPath)

    outputPath = os.path.join(basePath, 'output')
    if args.output:
        outputPath = os.path.join('', args.output)

    filename = ''
    if args.filename:
        filename = args.filename

    modelVersion = 'aparent'
    if args.modelVersion:
        modelVersion = args.modelVersion

    dataVersion = '1'
    if args.dataVersion:
        dataVersion = args.dataVersion

    sampleSize = None
    if args.sampleSize and args.sampleSize.isnumeric():
        sampleSize = int(args.sampleSize)
        print('Sample size: ' + args.sampleSize)

    print("BasePath: " + basePath)
    print("OutputPath: " + outputPath)
    print("DataPath: " + dataPath)
    print("filename: " + filename)
    print("modelVersion: " + modelVersion)
    print("dataVersion: " + dataVersion)
    return basePath, outputPath, dataPath, sampleSize, filename, dataVersion, modelVersion
