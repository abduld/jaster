module lib {
    export module example {
        export var mp1Source : string =
            "// MP 1\n\
#include    <wb.h>\n\
\n\
__global__ void vecAdd(float * in1, float * in2, float * out, int len) {\n\
    //@@ Insert code to implement vector addition here\n\
    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n\
    if (idx<len) out[idx] = in1[idx] + in2[idx];\n\
}\n\
\n\
int main(int argc, char ** argv) {\n\
    wbArg_t args;\n\
    int inputLength;\n\
    float * hostInput1;\n\
    float * hostInput2;\n\
    float * hostOutput;\n\
    float * deviceInput1;\n\
    float * deviceInput2;\n\
    float * deviceOutput;\n\
\n\
    args = wbArg_read(argc, argv);\n\
\n\
    wbTime_start(Generic, \"Importing data and creating memory on host\");\n\
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);\n\
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);\n\
    hostOutput = (float *) malloc(inputLength * sizeof(float));\n\
    wbTime_stop(Generic, \"Importing data and creating memory on host\");\n\
\n\
    wbLog(TRACE, \"The input length is \", inputLength, \" elements\");\n\
\n\
\n\
    wbTime_start(GPU, \"Allocating GPU memory.\");\n\
    //@@ Allocate GPU memory here\n\
    int byteSize =sizeof(float) * inputLength;\n\
\n\
    wbTime_stop(GPU, \"Allocating GPU memory.\");\n\
\n\
    wbTime_start(GPU, \"Copying input memory to the GPU.\");\n\
    //@@ Copy memory to the GPU here\n\
\n\
    cudaMalloc((void **) &deviceInput1, byteSize);\n\
    cudaMalloc((void **) &deviceInput2, byteSize);\n\
    cudaMalloc((void **) &deviceOutput, byteSize);\n\
\n\
\n\
    wbTime_stop(GPU, \"Copying input memory to the GPU.\");\n\
\n\
    //@@ Initialize the grid and block dimensions here\n\
    cudaMemcpy(deviceInput1, hostInput1, byteSize,cudaMemcpyHostToDevice);\n\
\n\
    cudaMemcpy(deviceInput2, hostInput1, byteSize,cudaMemcpyHostToDevice);\n\
\n\
\n\
    wbTime_start(Compute, \"Performing CUDA computation\");\n\
    //@@ Launch the GPU Kernel here\n\
    int block_size = 16;\n\
    int n_blocks = inputLength /block_size + (inputLength%block_size == 0 ? 0:1);\n\
\n\
\n\
    vecAdd<<< n_blocks, block_size>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);\n\
\n\
\n\
    cudaThreadSynchronize();\n\
    wbTime_stop(Compute, \"Performing CUDA computation\");\n\
\n\
    wbTime_start(Copy, \"Copying output memory to the CPU\");\n\
    //@@ Copy the GPU memory back to the CPU here\n\
    cudaMemcpy(hostOutput, deviceOutput, byteSize,cudaMemcpyDeviceToHost);\n\
\n\
    wbTime_stop(Copy, \"Copying output memory to the CPU\");\n\
\n\
    wbTime_start(GPU, \"Freeing GPU Memory\");\n\
    //@@ Free the GPU memory here\n\
\n\
\n\
    wbTime_stop(GPU, \"Freeing GPU Memory\");\n\
\n\
    wbSolution(args, hostOutput, inputLength);\n\
\n\
    free(hostInput1);\n\
    free(hostInput2);\n\
    free(hostOutput);\n\
\n\
    return 0;\n\
}";


        export var mp2Source : string =
            '\
            #include <wb.h>\n\
            \n\
            // Compute C = A * B\n\
            // Sgemm stands for single precision general matrix-matrix multiply\n\
                    __global__ void sgemm(float *A, float *B, float *C, int numARows,\n\
                    int numAColumns, int numBRows, int numBColumns) {\n\
                    //@@ Insert code to implement matrix multiplication here\n\
                    int row = blockIdx.y * blockDim.y + threadIdx.y;\n\
                    int col = blockIdx.x * blockDim.x + threadIdx.x;\n\
                    if (row < numARows && col < numBColumns) {\n\
                        float sum = 0;\n\
                        for (int ii = 0; ii < numAColumns; ii++) {\n\
                            sum += A[row * numAColumns + ii] * B[ii * numBColumns + col];\n\
                        }\n\
                        C[row * numBColumns + col] = sum;\n\
                    }\n\
                }\n\
            \n\
            #define wbCheck(stmt)\n\
            \n\
                    int main(int argc, char **argv) {\n\
                    wbArg_t args;\n\
                    float *hostA; // The A matrix\n\
                    float *hostB; // The B matrix\n\
                    float *hostC; // The output C matrix\n\
                    float *deviceA;\n\
                    float *deviceB;\n\
                    float *deviceC;\n\
                    int numARows;    // number of rows in the matrix A\n\
                    int numAColumns; // number of columns in the matrix A\n\
                    int numBRows;    // number of rows in the matrix B\n\
                    int numBColumns; // number of columns in the matrix B\n\
                    int numCRows;\n\
                    int numCColumns;\n\
            \n\
                    args = wbArg_read(argc, argv);\n\
            \n\
                    wbTime_start(Generic, "Importing data and creating memory on host");\n\
                    hostA =\n\
                        ( float * )wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);\n\
                    hostB =\n\
                        ( float * )wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);\n\
                    //@@ Allocate the hostC matrix\n\
                    hostC = ( float * )malloc(numARows * numBColumns * sizeof(float));\n\
                    wbTime_stop(Generic, "Importing data and creating memory on host");\n\
            \n\
                    numCRows = numARows;\n\
                    numCColumns = numBColumns;\n\
            \n\
                    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);\n\
                    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);\n\
                    wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);\n\
            \n\
                    wbTime_start(GPU, "Allocating GPU memory.");\n\
                    //@@ Allocate GPU memory here\n\
                    wbCheck(\n\
                        cudaMalloc(( void ** )&deviceA, numARows * numAColumns * sizeof(float)));\n\
                    wbCheck(\n\
                        cudaMalloc(( void ** )&deviceB, numBRows * numBColumns * sizeof(float)));\n\
                    wbCheck(\n\
                        cudaMalloc(( void ** )&deviceC, numARows * numBColumns * sizeof(float)));\n\
                    wbTime_stop(GPU, "Allocating GPU memory.");\n\
            \n\
                    wbTime_start(GPU, "Copying input memory to the GPU.");\n\
                    //@@ Copy memory to the GPU here\n\
                    wbCheck(cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float),\n\
                        cudaMemcpyHostToDevice));\n\
                    wbCheck(cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float),\n\
                        cudaMemcpyHostToDevice));\n\
                    wbTime_stop(GPU, "Copying input memory to the GPU.");\n\
            \n\
                    //@@ Initialize the grid and block dimensions here\n\
                    dim3 blockDim(16, 16);\n\
                    dim3 gridDim(ceil((( float )numAColumns) / blockDim.x),\n\
                    ceil((( float )numBRows) / blockDim.y));\n\
            \n\
                    wbLog(TRACE, "The block dimensions are ", blockDim.x, " x ", blockDim.y);\n\
                    wbLog(TRACE, "The grid dimensions are ", gridDim.x, " x ", gridDim.y);\n\
            \n\
                    wbTime_start(Compute, "Performing CUDA computation");\n\
                    //@@ Launch the GPU Kernel here\n\
                    wbCheck(cudaMemset(deviceC, 0, numARows * numBColumns * sizeof(float)));\n\
                    sgemm <<< gridDim, blockDim >>>\n\
                    (deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);\n\
                    cudaDeviceSynchronize();\n\
                    wbTime_stop(Compute, "Performing CUDA computation");\n\
            \n\
                    wbTime_start(Copy, "Copying output memory to the CPU");\n\
                    //@@ Copy the GPU memory back to the CPU here\n\
            \n\
                    wbCheck(cudaMemcpy(hostC, deviceC, numARows * numBColumns * sizeof(float),\n\
                        cudaMemcpyDeviceToHost));\n\
                    wbTime_stop(Copy, "Copying output memory to the CPU");\n\
            \n\
                    wbTime_start(GPU, "Freeing GPU Memory");\n\
                    //@@ Free the GPU memory here\n\
                    cudaFree(deviceA);\n\
                    cudaFree(deviceB);\n\
                    cudaFree(deviceC);\n\
                    wbTime_stop(GPU, "Freeing GPU Memory");\n\
            \n\
                    wbSolution(args, hostC, numARows, numBColumns);\n\
            \n\
                    free(hostA);\n\
                    free(hostB);\n\
                    free(hostC);\n\
            \n\
                    return 0;\n\
                }';
        export var mp1:any =
{
    "body": [
        {
            "attributes": [
                "__global__"
            ],
            "body": {
                "body": [
                    {
                        "cform": "int  idx = blockIdx.x * blockDim.x + threadIdx.x",
                        "declarations": [
                            {
                                "cform": "int  idx = blockIdx.x * blockDim.x + threadIdx.x",
                                "id": {
                                    "cform": "idx",
                                    "loc": {
                                        "end": {
                                            "column": 51,
                                            "line": 7
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 7
                                        }
                                    },
                                    "name": "idx",
                                    "raw": "int idx = blockIdx.x * blockDim.x + threadIdx.x",
                                    "type": "Identifier"
                                },
                                "init": {
                                    "cform": "blockIdx.x * blockDim.x + threadIdx.x",
                                    "left": {
                                        "cform": "blockIdx.x * blockDim.x",
                                        "left": {
                                            "cform": "blockIdx.x",
                                            "left": {
                                                "cform": "blockIdx",
                                                "kind": {
                                                    "address_spaces": [],
                                                    "bases": [
                                                        {
                                                            "cform": "uint3",
                                                            "loc": {
                                                                "end": {
                                                                    "column": 31,
                                                                    "line": 22
                                                                },
                                                                "start": {
                                                                    "column": 1,
                                                                    "line": 22
                                                                }
                                                            },
                                                            "raw": "uint3",
                                                            "type": "Literal",
                                                            "value": "uint3"
                                                        }
                                                    ],
                                                    "cform": "const uint3 ",
                                                    "loc": {
                                                        "end": {
                                                            "column": 31,
                                                            "line": 22
                                                        },
                                                        "start": {
                                                            "column": 1,
                                                            "line": 22
                                                        }
                                                    },
                                                    "qualifiers": [
                                                        {
                                                            "cform": "const",
                                                            "loc": {
                                                                "end": {
                                                                    "column": 31,
                                                                    "line": 22
                                                                },
                                                                "start": {
                                                                    "column": 1,
                                                                    "line": 22
                                                                }
                                                            },
                                                            "raw": "uint3 __device__ extern const blockIdx",
                                                            "type": "Literal",
                                                            "value": "const"
                                                        }
                                                    ],
                                                    "raw": "uint3 __device__ extern const blockIdx",
                                                    "type": "TypeSpecification"
                                                },
                                                "loc": {
                                                    "end": {
                                                        "column": 31,
                                                        "line": 22
                                                    },
                                                    "start": {
                                                        "column": 1,
                                                        "line": 22
                                                    }
                                                },
                                                "name": "blockIdx",
                                                "raw": "uint3 __device__ extern const blockIdx",
                                                "type": "Identifier"
                                            },
                                            "loc": {
                                                "end": {
                                                    "column": 24,
                                                    "line": 7
                                                },
                                                "start": {
                                                    "column": 15,
                                                    "line": 7
                                                }
                                            },
                                            "operator": ".",
                                            "raw": "blockIdx.x",
                                            "right": {
                                                "cform": "x",
                                                "kind": {
                                                    "address_spaces": [],
                                                    "bases": [
                                                        {
                                                            "cform": "unsigned int",
                                                            "loc": {
                                                                "end": {
                                                                    "column": 18,
                                                                    "line": 12
                                                                },
                                                                "start": {
                                                                    "column": 5,
                                                                    "line": 12
                                                                }
                                                            },
                                                            "raw": "unsigned int",
                                                            "type": "Literal",
                                                            "value": "unsigned int"
                                                        }
                                                    ],
                                                    "cform": "unsigned int ",
                                                    "loc": {
                                                        "end": {
                                                            "column": 18,
                                                            "line": 12
                                                        },
                                                        "start": {
                                                            "column": 5,
                                                            "line": 12
                                                        }
                                                    },
                                                    "qualifiers": [],
                                                    "raw": "unsigned int x",
                                                    "type": "TypeSpecification"
                                                },
                                                "loc": {
                                                    "end": {
                                                        "column": 18,
                                                        "line": 12
                                                    },
                                                    "start": {
                                                        "column": 5,
                                                        "line": 12
                                                    }
                                                },
                                                "name": "x",
                                                "raw": "unsigned int x",
                                                "type": "Identifier"
                                            },
                                            "type": "MemberExpression"
                                        },
                                        "loc": {
                                            "end": {
                                                "column": 37,
                                                "line": 7
                                            },
                                            "start": {
                                                "column": 15,
                                                "line": 7
                                            }
                                        },
                                        "operator": "*",
                                        "raw": "blockIdx.x * blockDim.x",
                                        "right": {
                                            "cform": "blockDim.x",
                                            "left": {
                                                "cform": "blockDim",
                                                "kind": {
                                                    "address_spaces": [],
                                                    "bases": [
                                                        {
                                                            "cform": "struct dim3",
                                                            "loc": {
                                                                "end": {
                                                                    "column": 30,
                                                                    "line": 23
                                                                },
                                                                "start": {
                                                                    "column": 1,
                                                                    "line": 23
                                                                }
                                                            },
                                                            "raw": "struct dim3",
                                                            "type": "Literal",
                                                            "value": "struct dim3"
                                                        }
                                                    ],
                                                    "cform": "const struct dim3 ",
                                                    "loc": {
                                                        "end": {
                                                            "column": 30,
                                                            "line": 23
                                                        },
                                                        "start": {
                                                            "column": 1,
                                                            "line": 23
                                                        }
                                                    },
                                                    "qualifiers": [
                                                        {
                                                            "cform": "const",
                                                            "loc": {
                                                                "end": {
                                                                    "column": 30,
                                                                    "line": 23
                                                                },
                                                                "start": {
                                                                    "column": 1,
                                                                    "line": 23
                                                                }
                                                            },
                                                            "raw": "dim3 __device__ extern const blockDim",
                                                            "type": "Literal",
                                                            "value": "const"
                                                        }
                                                    ],
                                                    "raw": "dim3 __device__ extern const blockDim",
                                                    "type": "TypeSpecification"
                                                },
                                                "loc": {
                                                    "end": {
                                                        "column": 30,
                                                        "line": 23
                                                    },
                                                    "start": {
                                                        "column": 1,
                                                        "line": 23
                                                    }
                                                },
                                                "name": "blockDim",
                                                "raw": "dim3 __device__ extern const blockDim",
                                                "type": "Identifier"
                                            },
                                            "loc": {
                                                "end": {
                                                    "column": 37,
                                                    "line": 7
                                                },
                                                "start": {
                                                    "column": 28,
                                                    "line": 7
                                                }
                                            },
                                            "operator": ".",
                                            "raw": "blockDim.x",
                                            "right": {
                                                "cform": "x",
                                                "kind": {
                                                    "address_spaces": [],
                                                    "bases": [
                                                        {
                                                            "cform": "unsigned int",
                                                            "loc": {
                                                                "end": {
                                                                    "column": 16,
                                                                    "line": 16
                                                                },
                                                                "start": {
                                                                    "column": 3,
                                                                    "line": 16
                                                                }
                                                            },
                                                            "raw": "unsigned int",
                                                            "type": "Literal",
                                                            "value": "unsigned int"
                                                        }
                                                    ],
                                                    "cform": "unsigned int ",
                                                    "loc": {
                                                        "end": {
                                                            "column": 16,
                                                            "line": 16
                                                        },
                                                        "start": {
                                                            "column": 3,
                                                            "line": 16
                                                        }
                                                    },
                                                    "qualifiers": [],
                                                    "raw": "unsigned int x",
                                                    "type": "TypeSpecification"
                                                },
                                                "loc": {
                                                    "end": {
                                                        "column": 16,
                                                        "line": 16
                                                    },
                                                    "start": {
                                                        "column": 3,
                                                        "line": 16
                                                    }
                                                },
                                                "name": "x",
                                                "raw": "unsigned int x",
                                                "type": "Identifier"
                                            },
                                            "type": "MemberExpression"
                                        },
                                        "type": "BinaryExpression"
                                    },
                                    "loc": {
                                        "end": {
                                            "column": 51,
                                            "line": 7
                                        },
                                        "start": {
                                            "column": 15,
                                            "line": 7
                                        }
                                    },
                                    "operator": "+",
                                    "raw": "blockIdx.x * blockDim.x + threadIdx.x",
                                    "right": {
                                        "cform": "threadIdx.x",
                                        "left": {
                                            "cform": "threadIdx",
                                            "kind": {
                                                "address_spaces": [],
                                                "bases": [
                                                    {
                                                        "cform": "uint3",
                                                        "loc": {
                                                            "end": {
                                                                "column": 31,
                                                                "line": 21
                                                            },
                                                            "start": {
                                                                "column": 1,
                                                                "line": 21
                                                            }
                                                        },
                                                        "raw": "uint3",
                                                        "type": "Literal",
                                                        "value": "uint3"
                                                    }
                                                ],
                                                "cform": "const uint3 ",
                                                "loc": {
                                                    "end": {
                                                        "column": 31,
                                                        "line": 21
                                                    },
                                                    "start": {
                                                        "column": 1,
                                                        "line": 21
                                                    }
                                                },
                                                "qualifiers": [
                                                    {
                                                        "cform": "const",
                                                        "loc": {
                                                            "end": {
                                                                "column": 31,
                                                                "line": 21
                                                            },
                                                            "start": {
                                                                "column": 1,
                                                                "line": 21
                                                            }
                                                        },
                                                        "raw": "uint3 __device__ extern const threadIdx",
                                                        "type": "Literal",
                                                        "value": "const"
                                                    }
                                                ],
                                                "raw": "uint3 __device__ extern const threadIdx",
                                                "type": "TypeSpecification"
                                            },
                                            "loc": {
                                                "end": {
                                                    "column": 31,
                                                    "line": 21
                                                },
                                                "start": {
                                                    "column": 1,
                                                    "line": 21
                                                }
                                            },
                                            "name": "threadIdx",
                                            "raw": "uint3 __device__ extern const threadIdx",
                                            "type": "Identifier"
                                        },
                                        "loc": {
                                            "end": {
                                                "column": 51,
                                                "line": 7
                                            },
                                            "start": {
                                                "column": 41,
                                                "line": 7
                                            }
                                        },
                                        "operator": ".",
                                        "raw": "threadIdx.x",
                                        "right": {
                                            "cform": "x",
                                            "kind": {
                                                "address_spaces": [],
                                                "bases": [
                                                    {
                                                        "cform": "unsigned int",
                                                        "loc": {
                                                            "end": {
                                                                "column": 18,
                                                                "line": 12
                                                            },
                                                            "start": {
                                                                "column": 5,
                                                                "line": 12
                                                            }
                                                        },
                                                        "raw": "unsigned int",
                                                        "type": "Literal",
                                                        "value": "unsigned int"
                                                    }
                                                ],
                                                "cform": "unsigned int ",
                                                "loc": {
                                                    "end": {
                                                        "column": 18,
                                                        "line": 12
                                                    },
                                                    "start": {
                                                        "column": 5,
                                                        "line": 12
                                                    }
                                                },
                                                "qualifiers": [],
                                                "raw": "unsigned int x",
                                                "type": "TypeSpecification"
                                            },
                                            "loc": {
                                                "end": {
                                                    "column": 18,
                                                    "line": 12
                                                },
                                                "start": {
                                                    "column": 5,
                                                    "line": 12
                                                }
                                            },
                                            "name": "x",
                                            "raw": "unsigned int x",
                                            "type": "Identifier"
                                        },
                                        "type": "MemberExpression"
                                    },
                                    "type": "BinaryExpression"
                                },
                                "kind": {
                                    "address_spaces": [],
                                    "bases": [
                                        {
                                            "cform": "int",
                                            "loc": {
                                                "end": {
                                                    "column": 51,
                                                    "line": 7
                                                },
                                                "start": {
                                                    "column": 5,
                                                    "line": 7
                                                }
                                            },
                                            "raw": "int",
                                            "type": "Literal",
                                            "value": "int"
                                        }
                                    ],
                                    "cform": "int ",
                                    "loc": {
                                        "end": {
                                            "column": 51,
                                            "line": 7
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 7
                                        }
                                    },
                                    "qualifiers": [],
                                    "raw": "int idx = blockIdx.x * blockDim.x + threadIdx.x",
                                    "type": "TypeSpecification"
                                },
                                "loc": {
                                    "end": {
                                        "column": 51,
                                        "line": 7
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 7
                                    }
                                },
                                "raw": "int idx = blockIdx.x * blockDim.x + threadIdx.x",
                                "type": "VariableDeclarator"
                            }
                        ],
                        "loc": {
                            "end": {
                                "column": 51,
                                "line": 7
                            },
                            "start": {
                                "column": 5,
                                "line": 7
                            }
                        },
                        "raw": "int idx = blockIdx.x * blockDim.x + threadIdx.x",
                        "type": "VariableDeclaration"
                    },
                    {
                        "cform": "if (idx < len){\nout[idx] = in1[idx] + in2[idx]; /* Assign*/\n}\n",
                        "consequent": {
                            "body": [
                                {
                                    "cform": "out[idx] = in1[idx] + in2[idx]",
                                    "left": {
                                        "cform": "out[idx]",
                                        "computed": true,
                                        "loc": {
                                            "end": {
                                                "column": 25,
                                                "line": 8
                                            },
                                            "start": {
                                                "column": 18,
                                                "line": 8
                                            }
                                        },
                                        "object": {
                                            "cform": "out",
                                            "kind": {
                                                "cform": "float *",
                                                "loc": {
                                                    "end": {
                                                        "column": 58,
                                                        "line": 5
                                                    },
                                                    "start": {
                                                        "column": 50,
                                                        "line": 5
                                                    }
                                                },
                                                "raw": "float * out",
                                                "type": "ReferenceType",
                                                "value": {
                                                    "address_spaces": [],
                                                    "bases": [
                                                        {
                                                            "cform": "float",
                                                            "loc": {
                                                                "end": {
                                                                    "column": 58,
                                                                    "line": 5
                                                                },
                                                                "start": {
                                                                    "column": 50,
                                                                    "line": 5
                                                                }
                                                            },
                                                            "raw": "float",
                                                            "type": "Literal",
                                                            "value": "float"
                                                        }
                                                    ],
                                                    "cform": "float ",
                                                    "loc": {
                                                        "end": {
                                                            "column": 58,
                                                            "line": 5
                                                        },
                                                        "start": {
                                                            "column": 50,
                                                            "line": 5
                                                        }
                                                    },
                                                    "qualifiers": [],
                                                    "raw": "float * out",
                                                    "type": "TypeSpecification"
                                                }
                                            },
                                            "loc": {
                                                "end": {
                                                    "column": 58,
                                                    "line": 5
                                                },
                                                "start": {
                                                    "column": 50,
                                                    "line": 5
                                                }
                                            },
                                            "name": "out",
                                            "raw": "float * out",
                                            "type": "Identifier"
                                        },
                                        "property": {
                                            "cform": "idx",
                                            "kind": {
                                                "address_spaces": [],
                                                "bases": [
                                                    {
                                                        "cform": "int",
                                                        "loc": {
                                                            "end": {
                                                                "column": 51,
                                                                "line": 7
                                                            },
                                                            "start": {
                                                                "column": 5,
                                                                "line": 7
                                                            }
                                                        },
                                                        "raw": "int",
                                                        "type": "Literal",
                                                        "value": "int"
                                                    }
                                                ],
                                                "cform": "int ",
                                                "loc": {
                                                    "end": {
                                                        "column": 51,
                                                        "line": 7
                                                    },
                                                    "start": {
                                                        "column": 5,
                                                        "line": 7
                                                    }
                                                },
                                                "qualifiers": [],
                                                "raw": "int idx = blockIdx.x * blockDim.x + threadIdx.x",
                                                "type": "TypeSpecification"
                                            },
                                            "loc": {
                                                "end": {
                                                    "column": 51,
                                                    "line": 7
                                                },
                                                "start": {
                                                    "column": 5,
                                                    "line": 7
                                                }
                                            },
                                            "name": "idx",
                                            "raw": "int idx = blockIdx.x * blockDim.x + threadIdx.x",
                                            "type": "Identifier"
                                        },
                                        "raw": "out[idx]",
                                        "type": "SubscriptExpression"
                                    },
                                    "loc": {
                                        "end": {
                                            "column": 47,
                                            "line": 8
                                        },
                                        "start": {
                                            "column": 18,
                                            "line": 8
                                        }
                                    },
                                    "operator": "=",
                                    "raw": "out[idx] = in1[idx] + in2[idx]",
                                    "right": {
                                        "cform": "in1[idx] + in2[idx]",
                                        "left": {
                                            "cform": "in1[idx]",
                                            "computed": true,
                                            "loc": {
                                                "end": {
                                                    "column": 36,
                                                    "line": 8
                                                },
                                                "start": {
                                                    "column": 29,
                                                    "line": 8
                                                }
                                            },
                                            "object": {
                                                "cform": "in1",
                                                "kind": {
                                                    "cform": "float *",
                                                    "loc": {
                                                        "end": {
                                                            "column": 32,
                                                            "line": 5
                                                        },
                                                        "start": {
                                                            "column": 24,
                                                            "line": 5
                                                        }
                                                    },
                                                    "raw": "float * in1",
                                                    "type": "ReferenceType",
                                                    "value": {
                                                        "address_spaces": [],
                                                        "bases": [
                                                            {
                                                                "cform": "float",
                                                                "loc": {
                                                                    "end": {
                                                                        "column": 32,
                                                                        "line": 5
                                                                    },
                                                                    "start": {
                                                                        "column": 24,
                                                                        "line": 5
                                                                    }
                                                                },
                                                                "raw": "float",
                                                                "type": "Literal",
                                                                "value": "float"
                                                            }
                                                        ],
                                                        "cform": "float ",
                                                        "loc": {
                                                            "end": {
                                                                "column": 32,
                                                                "line": 5
                                                            },
                                                            "start": {
                                                                "column": 24,
                                                                "line": 5
                                                            }
                                                        },
                                                        "qualifiers": [],
                                                        "raw": "float * in1",
                                                        "type": "TypeSpecification"
                                                    }
                                                },
                                                "loc": {
                                                    "end": {
                                                        "column": 32,
                                                        "line": 5
                                                    },
                                                    "start": {
                                                        "column": 24,
                                                        "line": 5
                                                    }
                                                },
                                                "name": "in1",
                                                "raw": "float * in1",
                                                "type": "Identifier"
                                            },
                                            "property": {
                                                "cform": "idx",
                                                "kind": {
                                                    "address_spaces": [],
                                                    "bases": [
                                                        {
                                                            "cform": "int",
                                                            "loc": {
                                                                "end": {
                                                                    "column": 51,
                                                                    "line": 7
                                                                },
                                                                "start": {
                                                                    "column": 5,
                                                                    "line": 7
                                                                }
                                                            },
                                                            "raw": "int",
                                                            "type": "Literal",
                                                            "value": "int"
                                                        }
                                                    ],
                                                    "cform": "int ",
                                                    "loc": {
                                                        "end": {
                                                            "column": 51,
                                                            "line": 7
                                                        },
                                                        "start": {
                                                            "column": 5,
                                                            "line": 7
                                                        }
                                                    },
                                                    "qualifiers": [],
                                                    "raw": "int idx = blockIdx.x * blockDim.x + threadIdx.x",
                                                    "type": "TypeSpecification"
                                                },
                                                "loc": {
                                                    "end": {
                                                        "column": 51,
                                                        "line": 7
                                                    },
                                                    "start": {
                                                        "column": 5,
                                                        "line": 7
                                                    }
                                                },
                                                "name": "idx",
                                                "raw": "int idx = blockIdx.x * blockDim.x + threadIdx.x",
                                                "type": "Identifier"
                                            },
                                            "raw": "in1[idx]",
                                            "type": "SubscriptExpression"
                                        },
                                        "loc": {
                                            "end": {
                                                "column": 47,
                                                "line": 8
                                            },
                                            "start": {
                                                "column": 29,
                                                "line": 8
                                            }
                                        },
                                        "operator": "+",
                                        "raw": "in1[idx] + in2[idx]",
                                        "right": {
                                            "cform": "in2[idx]",
                                            "computed": true,
                                            "loc": {
                                                "end": {
                                                    "column": 47,
                                                    "line": 8
                                                },
                                                "start": {
                                                    "column": 40,
                                                    "line": 8
                                                }
                                            },
                                            "object": {
                                                "cform": "in2",
                                                "kind": {
                                                    "cform": "float *",
                                                    "loc": {
                                                        "end": {
                                                            "column": 45,
                                                            "line": 5
                                                        },
                                                        "start": {
                                                            "column": 37,
                                                            "line": 5
                                                        }
                                                    },
                                                    "raw": "float * in2",
                                                    "type": "ReferenceType",
                                                    "value": {
                                                        "address_spaces": [],
                                                        "bases": [
                                                            {
                                                                "cform": "float",
                                                                "loc": {
                                                                    "end": {
                                                                        "column": 45,
                                                                        "line": 5
                                                                    },
                                                                    "start": {
                                                                        "column": 37,
                                                                        "line": 5
                                                                    }
                                                                },
                                                                "raw": "float",
                                                                "type": "Literal",
                                                                "value": "float"
                                                            }
                                                        ],
                                                        "cform": "float ",
                                                        "loc": {
                                                            "end": {
                                                                "column": 45,
                                                                "line": 5
                                                            },
                                                            "start": {
                                                                "column": 37,
                                                                "line": 5
                                                            }
                                                        },
                                                        "qualifiers": [],
                                                        "raw": "float * in2",
                                                        "type": "TypeSpecification"
                                                    }
                                                },
                                                "loc": {
                                                    "end": {
                                                        "column": 45,
                                                        "line": 5
                                                    },
                                                    "start": {
                                                        "column": 37,
                                                        "line": 5
                                                    }
                                                },
                                                "name": "in2",
                                                "raw": "float * in2",
                                                "type": "Identifier"
                                            },
                                            "property": {
                                                "cform": "idx",
                                                "kind": {
                                                    "address_spaces": [],
                                                    "bases": [
                                                        {
                                                            "cform": "int",
                                                            "loc": {
                                                                "end": {
                                                                    "column": 51,
                                                                    "line": 7
                                                                },
                                                                "start": {
                                                                    "column": 5,
                                                                    "line": 7
                                                                }
                                                            },
                                                            "raw": "int",
                                                            "type": "Literal",
                                                            "value": "int"
                                                        }
                                                    ],
                                                    "cform": "int ",
                                                    "loc": {
                                                        "end": {
                                                            "column": 51,
                                                            "line": 7
                                                        },
                                                        "start": {
                                                            "column": 5,
                                                            "line": 7
                                                        }
                                                    },
                                                    "qualifiers": [],
                                                    "raw": "int idx = blockIdx.x * blockDim.x + threadIdx.x",
                                                    "type": "TypeSpecification"
                                                },
                                                "loc": {
                                                    "end": {
                                                        "column": 51,
                                                        "line": 7
                                                    },
                                                    "start": {
                                                        "column": 5,
                                                        "line": 7
                                                    }
                                                },
                                                "name": "idx",
                                                "raw": "int idx = blockIdx.x * blockDim.x + threadIdx.x",
                                                "type": "Identifier"
                                            },
                                            "raw": "in2[idx]",
                                            "type": "SubscriptExpression"
                                        },
                                        "type": "BinaryExpression"
                                    },
                                    "type": "AssignmentExpression"
                                }
                            ],
                            "cform": "{\nout[idx] = in1[idx] + in2[idx]; /* Assign*/\n}\n",
                            "loc": {
                                "end": {
                                    "column": 47,
                                    "line": 8
                                },
                                "start": {
                                    "column": 5,
                                    "line": 8
                                }
                            },
                            "raw": "if (idx<len) out[idx] = in1[idx] + in2[idx]",
                            "type": "BlockStatement"
                        },
                        "loc": {
                            "end": {
                                "column": 47,
                                "line": 8
                            },
                            "start": {
                                "column": 5,
                                "line": 8
                            }
                        },
                        "raw": "if (idx<len) out[idx] = in1[idx] + in2[idx]",
                        "test": {
                            "cform": "idx < len",
                            "left": {
                                "cform": "idx",
                                "kind": {
                                    "address_spaces": [],
                                    "bases": [
                                        {
                                            "cform": "int",
                                            "loc": {
                                                "end": {
                                                    "column": 51,
                                                    "line": 7
                                                },
                                                "start": {
                                                    "column": 5,
                                                    "line": 7
                                                }
                                            },
                                            "raw": "int",
                                            "type": "Literal",
                                            "value": "int"
                                        }
                                    ],
                                    "cform": "int ",
                                    "loc": {
                                        "end": {
                                            "column": 51,
                                            "line": 7
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 7
                                        }
                                    },
                                    "qualifiers": [],
                                    "raw": "int idx = blockIdx.x * blockDim.x + threadIdx.x",
                                    "type": "TypeSpecification"
                                },
                                "loc": {
                                    "end": {
                                        "column": 51,
                                        "line": 7
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 7
                                    }
                                },
                                "name": "idx",
                                "raw": "int idx = blockIdx.x * blockDim.x + threadIdx.x",
                                "type": "Identifier"
                            },
                            "loc": {
                                "end": {
                                    "column": 13,
                                    "line": 8
                                },
                                "start": {
                                    "column": 9,
                                    "line": 8
                                }
                            },
                            "operator": "<",
                            "raw": "idx<len",
                            "right": {
                                "cform": "len",
                                "kind": {
                                    "address_spaces": [],
                                    "bases": [
                                        {
                                            "cform": "int",
                                            "loc": {
                                                "end": {
                                                    "column": 67,
                                                    "line": 5
                                                },
                                                "start": {
                                                    "column": 63,
                                                    "line": 5
                                                }
                                            },
                                            "raw": "int",
                                            "type": "Literal",
                                            "value": "int"
                                        }
                                    ],
                                    "cform": "int ",
                                    "loc": {
                                        "end": {
                                            "column": 67,
                                            "line": 5
                                        },
                                        "start": {
                                            "column": 63,
                                            "line": 5
                                        }
                                    },
                                    "qualifiers": [],
                                    "raw": "int len",
                                    "type": "TypeSpecification"
                                },
                                "loc": {
                                    "end": {
                                        "column": 67,
                                        "line": 5
                                    },
                                    "start": {
                                        "column": 63,
                                        "line": 5
                                    }
                                },
                                "name": "len",
                                "raw": "int len",
                                "type": "Identifier"
                            },
                            "type": "BinaryExpression"
                        },
                        "type": "IfStatement"
                    }
                ],
                "cform": "{\nint  idx = blockIdx.x * blockDim.x + threadIdx.x; /* Declare*/\nif (idx < len){\nout[idx] = in1[idx] + in2[idx]; /* Assign*/\n}\n}\n",
                "loc": {
                    "end": {
                        "column": 1,
                        "line": 9
                    },
                    "start": {
                        "column": 1,
                        "line": 5
                    }
                },
                "raw": "",
                "type": "BlockStatement"
            },
            "cform": "__global__ void  vecAdd(float * in1 /* Parameter*/, float * in2 /* Parameter*/, float * out /* Parameter*/, int  len){\nint  idx = blockIdx.x * blockDim.x + threadIdx.x; /* Declare*/\nif (idx < len){\nout[idx] = in1[idx] + in2[idx]; /* Assign*/\n}\n}\n",
            "id": "vecAdd",
            "loc": {
                "end": {
                    "column": 1,
                    "line": 9
                },
                "start": {
                    "column": 1,
                    "line": 5
                }
            },
            "params": [
                {
                    "data": {
                        "cform": "in1",
                        "loc": {
                            "end": {
                                "column": 32,
                                "line": 5
                            },
                            "start": {
                                "column": 24,
                                "line": 5
                            }
                        },
                        "name": "in1",
                        "raw": "float * in1",
                        "type": "Identifier"
                    },
                    "kind": {
                        "cform": "float *",
                        "loc": {
                            "end": {
                                "column": 32,
                                "line": 5
                            },
                            "start": {
                                "column": 24,
                                "line": 5
                            }
                        },
                        "raw": "float * in1",
                        "type": "ReferenceType",
                        "value": {
                            "address_spaces": [],
                            "bases": [
                                {
                                    "cform": "float",
                                    "loc": {
                                        "end": {
                                            "column": 32,
                                            "line": 5
                                        },
                                        "start": {
                                            "column": 24,
                                            "line": 5
                                        }
                                    },
                                    "raw": "float",
                                    "type": "Literal",
                                    "value": "float"
                                }
                            ],
                            "cform": "float ",
                            "loc": {
                                "end": {
                                    "column": 32,
                                    "line": 5
                                },
                                "start": {
                                    "column": 24,
                                    "line": 5
                                }
                            },
                            "qualifiers": [],
                            "raw": "float * in1",
                            "type": "TypeSpecification"
                        }
                    },
                    "type": "ParameterExpression"
                },
                {
                    "data": {
                        "cform": "in2",
                        "loc": {
                            "end": {
                                "column": 45,
                                "line": 5
                            },
                            "start": {
                                "column": 37,
                                "line": 5
                            }
                        },
                        "name": "in2",
                        "raw": "float * in2",
                        "type": "Identifier"
                    },
                    "kind": {
                        "cform": "float *",
                        "loc": {
                            "end": {
                                "column": 45,
                                "line": 5
                            },
                            "start": {
                                "column": 37,
                                "line": 5
                            }
                        },
                        "raw": "float * in2",
                        "type": "ReferenceType",
                        "value": {
                            "address_spaces": [],
                            "bases": [
                                {
                                    "cform": "float",
                                    "loc": {
                                        "end": {
                                            "column": 45,
                                            "line": 5
                                        },
                                        "start": {
                                            "column": 37,
                                            "line": 5
                                        }
                                    },
                                    "raw": "float",
                                    "type": "Literal",
                                    "value": "float"
                                }
                            ],
                            "cform": "float ",
                            "loc": {
                                "end": {
                                    "column": 45,
                                    "line": 5
                                },
                                "start": {
                                    "column": 37,
                                    "line": 5
                                }
                            },
                            "qualifiers": [],
                            "raw": "float * in2",
                            "type": "TypeSpecification"
                        }
                    },
                    "type": "ParameterExpression"
                },
                {
                    "data": {
                        "cform": "out",
                        "loc": {
                            "end": {
                                "column": 58,
                                "line": 5
                            },
                            "start": {
                                "column": 50,
                                "line": 5
                            }
                        },
                        "name": "out",
                        "raw": "float * out",
                        "type": "Identifier"
                    },
                    "kind": {
                        "cform": "float *",
                        "loc": {
                            "end": {
                                "column": 58,
                                "line": 5
                            },
                            "start": {
                                "column": 50,
                                "line": 5
                            }
                        },
                        "raw": "float * out",
                        "type": "ReferenceType",
                        "value": {
                            "address_spaces": [],
                            "bases": [
                                {
                                    "cform": "float",
                                    "loc": {
                                        "end": {
                                            "column": 58,
                                            "line": 5
                                        },
                                        "start": {
                                            "column": 50,
                                            "line": 5
                                        }
                                    },
                                    "raw": "float",
                                    "type": "Literal",
                                    "value": "float"
                                }
                            ],
                            "cform": "float ",
                            "loc": {
                                "end": {
                                    "column": 58,
                                    "line": 5
                                },
                                "start": {
                                    "column": 50,
                                    "line": 5
                                }
                            },
                            "qualifiers": [],
                            "raw": "float * out",
                            "type": "TypeSpecification"
                        }
                    },
                    "type": "ParameterExpression"
                },
                {
                    "data": {
                        "cform": "len",
                        "loc": {
                            "end": {
                                "column": 67,
                                "line": 5
                            },
                            "start": {
                                "column": 63,
                                "line": 5
                            }
                        },
                        "name": "len",
                        "raw": "int len",
                        "type": "Identifier"
                    },
                    "kind": {
                        "address_spaces": [],
                        "bases": [
                            {
                                "cform": "int",
                                "loc": {
                                    "end": {
                                        "column": 67,
                                        "line": 5
                                    },
                                    "start": {
                                        "column": 63,
                                        "line": 5
                                    }
                                },
                                "raw": "int",
                                "type": "Literal",
                                "value": "int"
                            }
                        ],
                        "cform": "int ",
                        "loc": {
                            "end": {
                                "column": 67,
                                "line": 5
                            },
                            "start": {
                                "column": 63,
                                "line": 5
                            }
                        },
                        "qualifiers": [],
                        "raw": "int len",
                        "type": "TypeSpecification"
                    },
                    "type": "ParameterExpression"
                }
            ],
            "raw": "",
            "type": "Function"
        },
        {
            "attributes": [],
            "body": {
                "body": [
                    {
                        "cform": "int  args",
                        "declarations": [
                            {
                                "cform": "int  args",
                                "id": {
                                    "cform": "args",
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 12
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 12
                                        }
                                    },
                                    "name": "args",
                                    "raw": "wbArg_t args",
                                    "type": "Identifier"
                                },
                                "kind": {
                                    "address_spaces": [],
                                    "bases": [
                                        {
                                            "cform": "int",
                                            "loc": {
                                                "end": {
                                                    "column": 13,
                                                    "line": 12
                                                },
                                                "start": {
                                                    "column": 5,
                                                    "line": 12
                                                }
                                            },
                                            "raw": "int",
                                            "type": "Literal",
                                            "value": "int"
                                        }
                                    ],
                                    "cform": "int ",
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 12
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 12
                                        }
                                    },
                                    "qualifiers": [],
                                    "raw": "wbArg_t args",
                                    "type": "TypeSpecification"
                                },
                                "loc": {
                                    "end": {
                                        "column": 13,
                                        "line": 12
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 12
                                    }
                                },
                                "raw": "wbArg_t args",
                                "type": "VariableDeclarator"
                            }
                        ],
                        "loc": {
                            "end": {
                                "column": 13,
                                "line": 12
                            },
                            "start": {
                                "column": 5,
                                "line": 12
                            }
                        },
                        "raw": "wbArg_t args",
                        "type": "VariableDeclaration"
                    },
                    {
                        "cform": "int  inputLength",
                        "declarations": [
                            {
                                "cform": "int  inputLength",
                                "id": {
                                    "cform": "inputLength",
                                    "loc": {
                                        "end": {
                                            "column": 9,
                                            "line": 13
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 13
                                        }
                                    },
                                    "name": "inputLength",
                                    "raw": "int inputLength",
                                    "type": "Identifier"
                                },
                                "kind": {
                                    "address_spaces": [],
                                    "bases": [
                                        {
                                            "cform": "int",
                                            "loc": {
                                                "end": {
                                                    "column": 9,
                                                    "line": 13
                                                },
                                                "start": {
                                                    "column": 5,
                                                    "line": 13
                                                }
                                            },
                                            "raw": "int",
                                            "type": "Literal",
                                            "value": "int"
                                        }
                                    ],
                                    "cform": "int ",
                                    "loc": {
                                        "end": {
                                            "column": 9,
                                            "line": 13
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 13
                                        }
                                    },
                                    "qualifiers": [],
                                    "raw": "int inputLength",
                                    "type": "TypeSpecification"
                                },
                                "loc": {
                                    "end": {
                                        "column": 9,
                                        "line": 13
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 13
                                    }
                                },
                                "raw": "int inputLength",
                                "type": "VariableDeclarator"
                            }
                        ],
                        "loc": {
                            "end": {
                                "column": 9,
                                "line": 13
                            },
                            "start": {
                                "column": 5,
                                "line": 13
                            }
                        },
                        "raw": "int inputLength",
                        "type": "VariableDeclaration"
                    },
                    {
                        "cform": "float * hostInput1",
                        "declarations": [
                            {
                                "cform": "float * hostInput1",
                                "id": {
                                    "cform": "hostInput1",
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 14
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 14
                                        }
                                    },
                                    "name": "hostInput1",
                                    "raw": "float * hostInput1",
                                    "type": "Identifier"
                                },
                                "kind": {
                                    "cform": "float *",
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 14
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 14
                                        }
                                    },
                                    "raw": "float * hostInput1",
                                    "type": "ReferenceType",
                                    "value": {
                                        "address_spaces": [],
                                        "bases": [
                                            {
                                                "cform": "float",
                                                "loc": {
                                                    "end": {
                                                        "column": 13,
                                                        "line": 14
                                                    },
                                                    "start": {
                                                        "column": 5,
                                                        "line": 14
                                                    }
                                                },
                                                "raw": "float",
                                                "type": "Literal",
                                                "value": "float"
                                            }
                                        ],
                                        "cform": "float ",
                                        "loc": {
                                            "end": {
                                                "column": 13,
                                                "line": 14
                                            },
                                            "start": {
                                                "column": 5,
                                                "line": 14
                                            }
                                        },
                                        "qualifiers": [],
                                        "raw": "float * hostInput1",
                                        "type": "TypeSpecification"
                                    }
                                },
                                "loc": {
                                    "end": {
                                        "column": 13,
                                        "line": 14
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 14
                                    }
                                },
                                "raw": "float * hostInput1",
                                "type": "VariableDeclarator"
                            }
                        ],
                        "loc": {
                            "end": {
                                "column": 13,
                                "line": 14
                            },
                            "start": {
                                "column": 5,
                                "line": 14
                            }
                        },
                        "raw": "float * hostInput1",
                        "type": "VariableDeclaration"
                    },
                    {
                        "cform": "float * hostInput2",
                        "declarations": [
                            {
                                "cform": "float * hostInput2",
                                "id": {
                                    "cform": "hostInput2",
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 15
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 15
                                        }
                                    },
                                    "name": "hostInput2",
                                    "raw": "float * hostInput2",
                                    "type": "Identifier"
                                },
                                "kind": {
                                    "cform": "float *",
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 15
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 15
                                        }
                                    },
                                    "raw": "float * hostInput2",
                                    "type": "ReferenceType",
                                    "value": {
                                        "address_spaces": [],
                                        "bases": [
                                            {
                                                "cform": "float",
                                                "loc": {
                                                    "end": {
                                                        "column": 13,
                                                        "line": 15
                                                    },
                                                    "start": {
                                                        "column": 5,
                                                        "line": 15
                                                    }
                                                },
                                                "raw": "float",
                                                "type": "Literal",
                                                "value": "float"
                                            }
                                        ],
                                        "cform": "float ",
                                        "loc": {
                                            "end": {
                                                "column": 13,
                                                "line": 15
                                            },
                                            "start": {
                                                "column": 5,
                                                "line": 15
                                            }
                                        },
                                        "qualifiers": [],
                                        "raw": "float * hostInput2",
                                        "type": "TypeSpecification"
                                    }
                                },
                                "loc": {
                                    "end": {
                                        "column": 13,
                                        "line": 15
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 15
                                    }
                                },
                                "raw": "float * hostInput2",
                                "type": "VariableDeclarator"
                            }
                        ],
                        "loc": {
                            "end": {
                                "column": 13,
                                "line": 15
                            },
                            "start": {
                                "column": 5,
                                "line": 15
                            }
                        },
                        "raw": "float * hostInput2",
                        "type": "VariableDeclaration"
                    },
                    {
                        "cform": "float * hostOutput",
                        "declarations": [
                            {
                                "cform": "float * hostOutput",
                                "id": {
                                    "cform": "hostOutput",
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 16
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 16
                                        }
                                    },
                                    "name": "hostOutput",
                                    "raw": "float * hostOutput",
                                    "type": "Identifier"
                                },
                                "kind": {
                                    "cform": "float *",
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 16
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 16
                                        }
                                    },
                                    "raw": "float * hostOutput",
                                    "type": "ReferenceType",
                                    "value": {
                                        "address_spaces": [],
                                        "bases": [
                                            {
                                                "cform": "float",
                                                "loc": {
                                                    "end": {
                                                        "column": 13,
                                                        "line": 16
                                                    },
                                                    "start": {
                                                        "column": 5,
                                                        "line": 16
                                                    }
                                                },
                                                "raw": "float",
                                                "type": "Literal",
                                                "value": "float"
                                            }
                                        ],
                                        "cform": "float ",
                                        "loc": {
                                            "end": {
                                                "column": 13,
                                                "line": 16
                                            },
                                            "start": {
                                                "column": 5,
                                                "line": 16
                                            }
                                        },
                                        "qualifiers": [],
                                        "raw": "float * hostOutput",
                                        "type": "TypeSpecification"
                                    }
                                },
                                "loc": {
                                    "end": {
                                        "column": 13,
                                        "line": 16
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 16
                                    }
                                },
                                "raw": "float * hostOutput",
                                "type": "VariableDeclarator"
                            }
                        ],
                        "loc": {
                            "end": {
                                "column": 13,
                                "line": 16
                            },
                            "start": {
                                "column": 5,
                                "line": 16
                            }
                        },
                        "raw": "float * hostOutput",
                        "type": "VariableDeclaration"
                    },
                    {
                        "cform": "float * deviceInput1",
                        "declarations": [
                            {
                                "cform": "float * deviceInput1",
                                "id": {
                                    "cform": "deviceInput1",
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 17
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 17
                                        }
                                    },
                                    "name": "deviceInput1",
                                    "raw": "float * deviceInput1",
                                    "type": "Identifier"
                                },
                                "kind": {
                                    "cform": "float *",
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 17
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 17
                                        }
                                    },
                                    "raw": "float * deviceInput1",
                                    "type": "ReferenceType",
                                    "value": {
                                        "address_spaces": [],
                                        "bases": [
                                            {
                                                "cform": "float",
                                                "loc": {
                                                    "end": {
                                                        "column": 13,
                                                        "line": 17
                                                    },
                                                    "start": {
                                                        "column": 5,
                                                        "line": 17
                                                    }
                                                },
                                                "raw": "float",
                                                "type": "Literal",
                                                "value": "float"
                                            }
                                        ],
                                        "cform": "float ",
                                        "loc": {
                                            "end": {
                                                "column": 13,
                                                "line": 17
                                            },
                                            "start": {
                                                "column": 5,
                                                "line": 17
                                            }
                                        },
                                        "qualifiers": [],
                                        "raw": "float * deviceInput1",
                                        "type": "TypeSpecification"
                                    }
                                },
                                "loc": {
                                    "end": {
                                        "column": 13,
                                        "line": 17
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 17
                                    }
                                },
                                "raw": "float * deviceInput1",
                                "type": "VariableDeclarator"
                            }
                        ],
                        "loc": {
                            "end": {
                                "column": 13,
                                "line": 17
                            },
                            "start": {
                                "column": 5,
                                "line": 17
                            }
                        },
                        "raw": "float * deviceInput1",
                        "type": "VariableDeclaration"
                    },
                    {
                        "cform": "float * deviceInput2",
                        "declarations": [
                            {
                                "cform": "float * deviceInput2",
                                "id": {
                                    "cform": "deviceInput2",
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 18
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 18
                                        }
                                    },
                                    "name": "deviceInput2",
                                    "raw": "float * deviceInput2",
                                    "type": "Identifier"
                                },
                                "kind": {
                                    "cform": "float *",
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 18
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 18
                                        }
                                    },
                                    "raw": "float * deviceInput2",
                                    "type": "ReferenceType",
                                    "value": {
                                        "address_spaces": [],
                                        "bases": [
                                            {
                                                "cform": "float",
                                                "loc": {
                                                    "end": {
                                                        "column": 13,
                                                        "line": 18
                                                    },
                                                    "start": {
                                                        "column": 5,
                                                        "line": 18
                                                    }
                                                },
                                                "raw": "float",
                                                "type": "Literal",
                                                "value": "float"
                                            }
                                        ],
                                        "cform": "float ",
                                        "loc": {
                                            "end": {
                                                "column": 13,
                                                "line": 18
                                            },
                                            "start": {
                                                "column": 5,
                                                "line": 18
                                            }
                                        },
                                        "qualifiers": [],
                                        "raw": "float * deviceInput2",
                                        "type": "TypeSpecification"
                                    }
                                },
                                "loc": {
                                    "end": {
                                        "column": 13,
                                        "line": 18
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 18
                                    }
                                },
                                "raw": "float * deviceInput2",
                                "type": "VariableDeclarator"
                            }
                        ],
                        "loc": {
                            "end": {
                                "column": 13,
                                "line": 18
                            },
                            "start": {
                                "column": 5,
                                "line": 18
                            }
                        },
                        "raw": "float * deviceInput2",
                        "type": "VariableDeclaration"
                    },
                    {
                        "cform": "float * deviceOutput",
                        "declarations": [
                            {
                                "cform": "float * deviceOutput",
                                "id": {
                                    "cform": "deviceOutput",
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 19
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 19
                                        }
                                    },
                                    "name": "deviceOutput",
                                    "raw": "float * deviceOutput",
                                    "type": "Identifier"
                                },
                                "kind": {
                                    "cform": "float *",
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 19
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 19
                                        }
                                    },
                                    "raw": "float * deviceOutput",
                                    "type": "ReferenceType",
                                    "value": {
                                        "address_spaces": [],
                                        "bases": [
                                            {
                                                "cform": "float",
                                                "loc": {
                                                    "end": {
                                                        "column": 13,
                                                        "line": 19
                                                    },
                                                    "start": {
                                                        "column": 5,
                                                        "line": 19
                                                    }
                                                },
                                                "raw": "float",
                                                "type": "Literal",
                                                "value": "float"
                                            }
                                        ],
                                        "cform": "float ",
                                        "loc": {
                                            "end": {
                                                "column": 13,
                                                "line": 19
                                            },
                                            "start": {
                                                "column": 5,
                                                "line": 19
                                            }
                                        },
                                        "qualifiers": [],
                                        "raw": "float * deviceOutput",
                                        "type": "TypeSpecification"
                                    }
                                },
                                "loc": {
                                    "end": {
                                        "column": 13,
                                        "line": 19
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 19
                                    }
                                },
                                "raw": "float * deviceOutput",
                                "type": "VariableDeclarator"
                            }
                        ],
                        "loc": {
                            "end": {
                                "column": 13,
                                "line": 19
                            },
                            "start": {
                                "column": 5,
                                "line": 19
                            }
                        },
                        "raw": "float * deviceOutput",
                        "type": "VariableDeclaration"
                    },
                    {
                        "cform": "args = wbArg_read(argc /* Identifier*/, argv)",
                        "left": {
                            "cform": "args",
                            "kind": {
                                "address_spaces": [],
                                "bases": [
                                    {
                                        "cform": "int",
                                        "loc": {
                                            "end": {
                                                "column": 13,
                                                "line": 12
                                            },
                                            "start": {
                                                "column": 5,
                                                "line": 12
                                            }
                                        },
                                        "raw": "int",
                                        "type": "Literal",
                                        "value": "int"
                                    }
                                ],
                                "cform": "int ",
                                "loc": {
                                    "end": {
                                        "column": 13,
                                        "line": 12
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 12
                                    }
                                },
                                "qualifiers": [],
                                "raw": "wbArg_t args",
                                "type": "TypeSpecification"
                            },
                            "loc": {
                                "end": {
                                    "column": 13,
                                    "line": 12
                                },
                                "start": {
                                    "column": 5,
                                    "line": 12
                                }
                            },
                            "name": "args",
                            "raw": "wbArg_t args",
                            "type": "Identifier"
                        },
                        "loc": {
                            "end": {
                                "column": 33,
                                "line": 21
                            },
                            "start": {
                                "column": 5,
                                "line": 21
                            }
                        },
                        "operator": "=",
                        "raw": "args = wbArg_read(argc, argv)",
                        "right": {
                            "arguments": [
                                {
                                    "cform": "argc",
                                    "kind": {
                                        "address_spaces": [],
                                        "bases": [
                                            {
                                                "cform": "int",
                                                "loc": {
                                                    "end": {
                                                        "column": 14,
                                                        "line": 11
                                                    },
                                                    "start": {
                                                        "column": 10,
                                                        "line": 11
                                                    }
                                                },
                                                "raw": "int",
                                                "type": "Literal",
                                                "value": "int"
                                            }
                                        ],
                                        "cform": "int ",
                                        "loc": {
                                            "end": {
                                                "column": 14,
                                                "line": 11
                                            },
                                            "start": {
                                                "column": 10,
                                                "line": 11
                                            }
                                        },
                                        "qualifiers": [],
                                        "raw": "int argc",
                                        "type": "TypeSpecification"
                                    },
                                    "loc": {
                                        "end": {
                                            "column": 14,
                                            "line": 11
                                        },
                                        "start": {
                                            "column": 10,
                                            "line": 11
                                        }
                                    },
                                    "name": "argc",
                                    "raw": "int argc",
                                    "type": "Identifier"
                                },
                                {
                                    "cform": "argv",
                                    "kind": {
                                        "cform": "char **",
                                        "loc": {
                                            "end": {
                                                "column": 28,
                                                "line": 11
                                            },
                                            "start": {
                                                "column": 20,
                                                "line": 11
                                            }
                                        },
                                        "raw": "char ** argv",
                                        "type": "ReferenceType",
                                        "value": {
                                            "cform": "char *",
                                            "loc": {
                                                "end": {
                                                    "column": 28,
                                                    "line": 11
                                                },
                                                "start": {
                                                    "column": 20,
                                                    "line": 11
                                                }
                                            },
                                            "raw": "char ** argv",
                                            "type": "ReferenceType",
                                            "value": {
                                                "address_spaces": [],
                                                "bases": [
                                                    {
                                                        "cform": "char",
                                                        "loc": {
                                                            "end": {
                                                                "column": 28,
                                                                "line": 11
                                                            },
                                                            "start": {
                                                                "column": 20,
                                                                "line": 11
                                                            }
                                                        },
                                                        "raw": "char",
                                                        "type": "Literal",
                                                        "value": "char"
                                                    }
                                                ],
                                                "cform": "char ",
                                                "loc": {
                                                    "end": {
                                                        "column": 28,
                                                        "line": 11
                                                    },
                                                    "start": {
                                                        "column": 20,
                                                        "line": 11
                                                    }
                                                },
                                                "qualifiers": [],
                                                "raw": "char ** argv",
                                                "type": "TypeSpecification"
                                            }
                                        }
                                    },
                                    "loc": {
                                        "end": {
                                            "column": 28,
                                            "line": 11
                                        },
                                        "start": {
                                            "column": 20,
                                            "line": 11
                                        }
                                    },
                                    "name": "argv",
                                    "raw": "char ** argv",
                                    "type": "Identifier"
                                }
                            ],
                            "callee": {
                                "cform": "wbArg_read",
                                "loc": {
                                    "end": {
                                        "column": 33,
                                        "line": 21
                                    },
                                    "start": {
                                        "column": 12,
                                        "line": 21
                                    }
                                },
                                "name": "wbArg_read",
                                "raw": "wbArg_read(argc, argv)",
                                "type": "Identifier"
                            },
                            "cform": "wbArg_read(argc /* Identifier*/, argv)",
                            "loc": {
                                "end": {
                                    "column": 33,
                                    "line": 21
                                },
                                "start": {
                                    "column": 12,
                                    "line": 21
                                }
                            },
                            "raw": "wbArg_read(argc, argv)",
                            "type": "CallExpression"
                        },
                        "type": "AssignmentExpression"
                    },
                    {
                        "arguments": [
                            {
                                "cform": "\"Generic\"",
                                "loc": {
                                    "end": {
                                        "column": 5,
                                        "line": 23
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 23
                                    }
                                },
                                "raw": "Generic",
                                "type": "StringLiteral",
                                "value": "\"Generic\""
                            },
                            {
                                "cform": "\"Importing data and creating memory on host\"",
                                "loc": {
                                    "end": {
                                        "column": 5,
                                        "line": 23
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 23
                                    }
                                },
                                "raw": "Importing data and creating memory on host",
                                "type": "StringLiteral",
                                "value": "\"Importing data and creating memory on host\""
                            }
                        ],
                        "callee": {
                            "cform": "wbTime_start",
                            "loc": {
                                "end": {
                                    "column": 5,
                                    "line": 23
                                },
                                "start": {
                                    "column": 5,
                                    "line": 23
                                }
                            },
                            "name": "wbTime_start",
                            "raw": "wbTime_start(#kind, __VA_ARGS__)",
                            "type": "Identifier"
                        },
                        "cform": "wbTime_start(\"Generic\" /* String*/, \"Importing data and creating memory on host\")",
                        "loc": {
                            "end": {
                                "column": 5,
                                "line": 23
                            },
                            "start": {
                                "column": 5,
                                "line": 23
                            }
                        },
                        "raw": "wbTime_start(#kind, __VA_ARGS__)",
                        "type": "CallExpression"
                    },
                    {},
                    {
                        "cform": "hostInput1 = wbImport(\"input0\" /* String*/, & inputLength)",
                        "left": {
                            "cform": "hostInput1",
                            "kind": {
                                "cform": "float *",
                                "loc": {
                                    "end": {
                                        "column": 13,
                                        "line": 14
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 14
                                    }
                                },
                                "raw": "float * hostInput1",
                                "type": "ReferenceType",
                                "value": {
                                    "address_spaces": [],
                                    "bases": [
                                        {
                                            "cform": "float",
                                            "loc": {
                                                "end": {
                                                    "column": 13,
                                                    "line": 14
                                                },
                                                "start": {
                                                    "column": 5,
                                                    "line": 14
                                                }
                                            },
                                            "raw": "float",
                                            "type": "Literal",
                                            "value": "float"
                                        }
                                    ],
                                    "cform": "float ",
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 14
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 14
                                        }
                                    },
                                    "qualifiers": [],
                                    "raw": "float * hostInput1",
                                    "type": "TypeSpecification"
                                }
                            },
                            "loc": {
                                "end": {
                                    "column": 13,
                                    "line": 14
                                },
                                "start": {
                                    "column": 5,
                                    "line": 14
                                }
                            },
                            "name": "hostInput1",
                            "raw": "float * hostInput1",
                            "type": "Identifier"
                        },
                        "loc": {
                            "end": {
                                "column": 78,
                                "line": 24
                            },
                            "start": {
                                "column": 5,
                                "line": 24
                            }
                        },
                        "operator": "=",
                        "raw": "hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength)",
                        "right": {
                            "arguments": [
                                {
                                    "cform": "\"input0\"",
                                    "loc": {
                                        "end": {
                                            "column": 37,
                                            "line": 24
                                        },
                                        "start": {
                                            "column": 37,
                                            "line": 24
                                        }
                                    },
                                    "raw": "input0",
                                    "type": "StringLiteral",
                                    "value": "\"input0\""
                                },
                                {
                                    "argument": {
                                        "cform": "inputLength",
                                        "kind": {
                                            "address_spaces": [],
                                            "bases": [
                                                {
                                                    "cform": "int",
                                                    "loc": {
                                                        "end": {
                                                            "column": 9,
                                                            "line": 13
                                                        },
                                                        "start": {
                                                            "column": 5,
                                                            "line": 13
                                                        }
                                                    },
                                                    "raw": "int",
                                                    "type": "Literal",
                                                    "value": "int"
                                                }
                                            ],
                                            "cform": "int ",
                                            "loc": {
                                                "end": {
                                                    "column": 9,
                                                    "line": 13
                                                },
                                                "start": {
                                                    "column": 5,
                                                    "line": 13
                                                }
                                            },
                                            "qualifiers": [],
                                            "raw": "int inputLength",
                                            "type": "TypeSpecification"
                                        },
                                        "loc": {
                                            "end": {
                                                "column": 9,
                                                "line": 13
                                            },
                                            "start": {
                                                "column": 5,
                                                "line": 13
                                            }
                                        },
                                        "name": "inputLength",
                                        "raw": "int inputLength",
                                        "type": "Identifier"
                                    },
                                    "cform": "& inputLength",
                                    "loc": {
                                        "end": {
                                            "column": 67,
                                            "line": 24
                                        },
                                        "start": {
                                            "column": 66,
                                            "line": 24
                                        }
                                    },
                                    "operator": "&",
                                    "prefix": true,
                                    "raw": "&inputLength",
                                    "type": "UnaryExpression"
                                }
                            ],
                            "callee": {
                                "cform": "wbImport",
                                "loc": {
                                    "end": {
                                        "column": 78,
                                        "line": 24
                                    },
                                    "start": {
                                        "column": 28,
                                        "line": 24
                                    }
                                },
                                "name": "wbImport",
                                "raw": "wbImport(wbArg_getInputFile(args, 0), &inputLength)",
                                "type": "Identifier"
                            },
                            "cform": "wbImport(\"input0\" /* String*/, & inputLength)",
                            "loc": {
                                "end": {
                                    "column": 78,
                                    "line": 24
                                },
                                "start": {
                                    "column": 28,
                                    "line": 24
                                }
                            },
                            "raw": "wbImport(wbArg_getInputFile(args, 0), &inputLength)",
                            "type": "CallExpression"
                        },
                        "type": "AssignmentExpression"
                    },
                    {
                        "cform": "hostInput2 = wbImport(\"input1\" /* String*/, & inputLength)",
                        "left": {
                            "cform": "hostInput2",
                            "kind": {
                                "cform": "float *",
                                "loc": {
                                    "end": {
                                        "column": 13,
                                        "line": 15
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 15
                                    }
                                },
                                "raw": "float * hostInput2",
                                "type": "ReferenceType",
                                "value": {
                                    "address_spaces": [],
                                    "bases": [
                                        {
                                            "cform": "float",
                                            "loc": {
                                                "end": {
                                                    "column": 13,
                                                    "line": 15
                                                },
                                                "start": {
                                                    "column": 5,
                                                    "line": 15
                                                }
                                            },
                                            "raw": "float",
                                            "type": "Literal",
                                            "value": "float"
                                        }
                                    ],
                                    "cform": "float ",
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 15
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 15
                                        }
                                    },
                                    "qualifiers": [],
                                    "raw": "float * hostInput2",
                                    "type": "TypeSpecification"
                                }
                            },
                            "loc": {
                                "end": {
                                    "column": 13,
                                    "line": 15
                                },
                                "start": {
                                    "column": 5,
                                    "line": 15
                                }
                            },
                            "name": "hostInput2",
                            "raw": "float * hostInput2",
                            "type": "Identifier"
                        },
                        "loc": {
                            "end": {
                                "column": 78,
                                "line": 25
                            },
                            "start": {
                                "column": 5,
                                "line": 25
                            }
                        },
                        "operator": "=",
                        "raw": "hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength)",
                        "right": {
                            "arguments": [
                                {
                                    "cform": "\"input1\"",
                                    "loc": {
                                        "end": {
                                            "column": 37,
                                            "line": 25
                                        },
                                        "start": {
                                            "column": 37,
                                            "line": 25
                                        }
                                    },
                                    "raw": "input1",
                                    "type": "StringLiteral",
                                    "value": "\"input1\""
                                },
                                {
                                    "argument": {
                                        "cform": "inputLength",
                                        "kind": {
                                            "address_spaces": [],
                                            "bases": [
                                                {
                                                    "cform": "int",
                                                    "loc": {
                                                        "end": {
                                                            "column": 9,
                                                            "line": 13
                                                        },
                                                        "start": {
                                                            "column": 5,
                                                            "line": 13
                                                        }
                                                    },
                                                    "raw": "int",
                                                    "type": "Literal",
                                                    "value": "int"
                                                }
                                            ],
                                            "cform": "int ",
                                            "loc": {
                                                "end": {
                                                    "column": 9,
                                                    "line": 13
                                                },
                                                "start": {
                                                    "column": 5,
                                                    "line": 13
                                                }
                                            },
                                            "qualifiers": [],
                                            "raw": "int inputLength",
                                            "type": "TypeSpecification"
                                        },
                                        "loc": {
                                            "end": {
                                                "column": 9,
                                                "line": 13
                                            },
                                            "start": {
                                                "column": 5,
                                                "line": 13
                                            }
                                        },
                                        "name": "inputLength",
                                        "raw": "int inputLength",
                                        "type": "Identifier"
                                    },
                                    "cform": "& inputLength",
                                    "loc": {
                                        "end": {
                                            "column": 67,
                                            "line": 25
                                        },
                                        "start": {
                                            "column": 66,
                                            "line": 25
                                        }
                                    },
                                    "operator": "&",
                                    "prefix": true,
                                    "raw": "&inputLength",
                                    "type": "UnaryExpression"
                                }
                            ],
                            "callee": {
                                "cform": "wbImport",
                                "loc": {
                                    "end": {
                                        "column": 78,
                                        "line": 25
                                    },
                                    "start": {
                                        "column": 28,
                                        "line": 25
                                    }
                                },
                                "name": "wbImport",
                                "raw": "wbImport(wbArg_getInputFile(args, 1), &inputLength)",
                                "type": "Identifier"
                            },
                            "cform": "wbImport(\"input1\" /* String*/, & inputLength)",
                            "loc": {
                                "end": {
                                    "column": 78,
                                    "line": 25
                                },
                                "start": {
                                    "column": 28,
                                    "line": 25
                                }
                            },
                            "raw": "wbImport(wbArg_getInputFile(args, 1), &inputLength)",
                            "type": "CallExpression"
                        },
                        "type": "AssignmentExpression"
                    },
                    {
                        "cform": "hostOutput = malloc(inputLength * sizeof(float ))",
                        "left": {
                            "cform": "hostOutput",
                            "kind": {
                                "cform": "float *",
                                "loc": {
                                    "end": {
                                        "column": 13,
                                        "line": 16
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 16
                                    }
                                },
                                "raw": "float * hostOutput",
                                "type": "ReferenceType",
                                "value": {
                                    "address_spaces": [],
                                    "bases": [
                                        {
                                            "cform": "float",
                                            "loc": {
                                                "end": {
                                                    "column": 13,
                                                    "line": 16
                                                },
                                                "start": {
                                                    "column": 5,
                                                    "line": 16
                                                }
                                            },
                                            "raw": "float",
                                            "type": "Literal",
                                            "value": "float"
                                        }
                                    ],
                                    "cform": "float ",
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 16
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 16
                                        }
                                    },
                                    "qualifiers": [],
                                    "raw": "float * hostOutput",
                                    "type": "TypeSpecification"
                                }
                            },
                            "loc": {
                                "end": {
                                    "column": 13,
                                    "line": 16
                                },
                                "start": {
                                    "column": 5,
                                    "line": 16
                                }
                            },
                            "name": "hostOutput",
                            "raw": "float * hostOutput",
                            "type": "Identifier"
                        },
                        "loc": {
                            "end": {
                                "column": 62,
                                "line": 26
                            },
                            "start": {
                                "column": 5,
                                "line": 26
                            }
                        },
                        "operator": "=",
                        "raw": "hostOutput = (float *) malloc(inputLength * sizeof(float))",
                        "right": {
                            "arguments": [
                                {
                                    "cform": "inputLength * sizeof(float )",
                                    "left": {
                                        "cform": "inputLength",
                                        "kind": {
                                            "address_spaces": [],
                                            "bases": [
                                                {
                                                    "cform": "int",
                                                    "loc": {
                                                        "end": {
                                                            "column": 9,
                                                            "line": 13
                                                        },
                                                        "start": {
                                                            "column": 5,
                                                            "line": 13
                                                        }
                                                    },
                                                    "raw": "int",
                                                    "type": "Literal",
                                                    "value": "int"
                                                }
                                            ],
                                            "cform": "int ",
                                            "loc": {
                                                "end": {
                                                    "column": 9,
                                                    "line": 13
                                                },
                                                "start": {
                                                    "column": 5,
                                                    "line": 13
                                                }
                                            },
                                            "qualifiers": [],
                                            "raw": "int inputLength",
                                            "type": "TypeSpecification"
                                        },
                                        "loc": {
                                            "end": {
                                                "column": 9,
                                                "line": 13
                                            },
                                            "start": {
                                                "column": 5,
                                                "line": 13
                                            }
                                        },
                                        "name": "inputLength",
                                        "raw": "int inputLength",
                                        "type": "Identifier"
                                    },
                                    "loc": {
                                        "end": {
                                            "column": 61,
                                            "line": 26
                                        },
                                        "start": {
                                            "column": 35,
                                            "line": 26
                                        }
                                    },
                                    "operator": "*",
                                    "raw": "inputLength * sizeof(float)",
                                    "right": {
                                        "arguments": [
                                            {
                                                "address_spaces": [],
                                                "bases": [
                                                    {
                                                        "cform": "float",
                                                        "loc": {
                                                            "end": {
                                                                "column": 61,
                                                                "line": 26
                                                            },
                                                            "start": {
                                                                "column": 49,
                                                                "line": 26
                                                            }
                                                        },
                                                        "raw": "float",
                                                        "type": "Literal",
                                                        "value": "float"
                                                    }
                                                ],
                                                "cform": "float ",
                                                "loc": {
                                                    "end": {
                                                        "column": 61,
                                                        "line": 26
                                                    },
                                                    "start": {
                                                        "column": 49,
                                                        "line": 26
                                                    }
                                                },
                                                "qualifiers": [],
                                                "raw": "sizeof(float)",
                                                "type": "TypeSpecification"
                                            }
                                        ],
                                        "callee": {
                                            "cform": "sizeof",
                                            "loc": {
                                                "end": {
                                                    "column": 61,
                                                    "line": 26
                                                },
                                                "start": {
                                                    "column": 49,
                                                    "line": 26
                                                }
                                            },
                                            "name": "sizeof",
                                            "raw": "sizeof(float)",
                                            "type": "Identifier"
                                        },
                                        "cform": "sizeof(float )",
                                        "loc": {
                                            "end": {
                                                "column": 61,
                                                "line": 26
                                            },
                                            "start": {
                                                "column": 49,
                                                "line": 26
                                            }
                                        },
                                        "raw": "sizeof(float)",
                                        "type": "CallExpression"
                                    },
                                    "type": "BinaryExpression"
                                }
                            ],
                            "callee": {
                                "cform": "malloc",
                                "loc": {
                                    "end": {
                                        "column": 62,
                                        "line": 26
                                    },
                                    "start": {
                                        "column": 28,
                                        "line": 26
                                    }
                                },
                                "name": "malloc",
                                "raw": "malloc(inputLength * sizeof(float))",
                                "type": "Identifier"
                            },
                            "cform": "malloc(inputLength * sizeof(float ))",
                            "loc": {
                                "end": {
                                    "column": 62,
                                    "line": 26
                                },
                                "start": {
                                    "column": 28,
                                    "line": 26
                                }
                            },
                            "raw": "malloc(inputLength * sizeof(float))",
                            "type": "CallExpression"
                        },
                        "type": "AssignmentExpression"
                    },
                    {
                        "arguments": [
                            {
                                "cform": "\"Generic\"",
                                "loc": {
                                    "end": {
                                        "column": 5,
                                        "line": 27
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 27
                                    }
                                },
                                "raw": "Generic",
                                "type": "StringLiteral",
                                "value": "\"Generic\""
                            },
                            {
                                "cform": "\"Importing data and creating memory on host\"",
                                "loc": {
                                    "end": {
                                        "column": 5,
                                        "line": 27
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 27
                                    }
                                },
                                "raw": "Importing data and creating memory on host",
                                "type": "StringLiteral",
                                "value": "\"Importing data and creating memory on host\""
                            }
                        ],
                        "callee": {
                            "cform": "wbTime_stop",
                            "loc": {
                                "end": {
                                    "column": 5,
                                    "line": 27
                                },
                                "start": {
                                    "column": 5,
                                    "line": 27
                                }
                            },
                            "name": "wbTime_stop",
                            "raw": "wbTime_stop(#kind, __VA_ARGS__)",
                            "type": "Identifier"
                        },
                        "cform": "wbTime_stop(\"Generic\" /* String*/, \"Importing data and creating memory on host\")",
                        "loc": {
                            "end": {
                                "column": 5,
                                "line": 27
                            },
                            "start": {
                                "column": 5,
                                "line": 27
                            }
                        },
                        "raw": "wbTime_stop(#kind, __VA_ARGS__)",
                        "type": "CallExpression"
                    },
                    {},
                    {
                        "arguments": [
                            {
                                "cform": "\"TRACE\"",
                                "loc": {
                                    "end": {
                                        "column": 5,
                                        "line": 29
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 29
                                    }
                                },
                                "raw": "TRACE",
                                "type": "StringLiteral",
                                "value": "\"TRACE\""
                            },
                            {
                                "cform": "\"The input length is \"",
                                "loc": {
                                    "end": {
                                        "column": 5,
                                        "line": 29
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 29
                                    }
                                },
                                "raw": "The input length is ",
                                "type": "StringLiteral",
                                "value": "\"The input length is \""
                            },
                            {
                                "cform": "inputLength",
                                "kind": {
                                    "address_spaces": [],
                                    "bases": [
                                        {
                                            "cform": "int",
                                            "loc": {
                                                "end": {
                                                    "column": 9,
                                                    "line": 13
                                                },
                                                "start": {
                                                    "column": 5,
                                                    "line": 13
                                                }
                                            },
                                            "raw": "int",
                                            "type": "Literal",
                                            "value": "int"
                                        }
                                    ],
                                    "cform": "int ",
                                    "loc": {
                                        "end": {
                                            "column": 9,
                                            "line": 13
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 13
                                        }
                                    },
                                    "qualifiers": [],
                                    "raw": "int inputLength",
                                    "type": "TypeSpecification"
                                },
                                "loc": {
                                    "end": {
                                        "column": 9,
                                        "line": 13
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 13
                                    }
                                },
                                "name": "inputLength",
                                "raw": "int inputLength",
                                "type": "Identifier"
                            },
                            {
                                "cform": "\" elements\"",
                                "loc": {
                                    "end": {
                                        "column": 5,
                                        "line": 29
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 29
                                    }
                                },
                                "raw": " elements",
                                "type": "StringLiteral",
                                "value": "\" elements\""
                            }
                        ],
                        "callee": {
                            "cform": "wbLog",
                            "loc": {
                                "end": {
                                    "column": 5,
                                    "line": 29
                                },
                                "start": {
                                    "column": 5,
                                    "line": 29
                                }
                            },
                            "name": "wbLog",
                            "raw": "wbLog(#level, __VA_ARGS__)",
                            "type": "Identifier"
                        },
                        "cform": "wbLog(\"TRACE\" /* String*/, \"The input length is \" /* String*/, inputLength /* Identifier*/, \" elements\")",
                        "loc": {
                            "end": {
                                "column": 5,
                                "line": 29
                            },
                            "start": {
                                "column": 5,
                                "line": 29
                            }
                        },
                        "raw": "wbLog(#level, __VA_ARGS__)",
                        "type": "CallExpression"
                    },
                    {
                        "arguments": [
                            {
                                "cform": "\"GPU\"",
                                "loc": {
                                    "end": {
                                        "column": 5,
                                        "line": 32
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 32
                                    }
                                },
                                "raw": "GPU",
                                "type": "StringLiteral",
                                "value": "\"GPU\""
                            },
                            {
                                "cform": "\"Allocating GPU memory.\"",
                                "loc": {
                                    "end": {
                                        "column": 5,
                                        "line": 32
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 32
                                    }
                                },
                                "raw": "Allocating GPU memory.",
                                "type": "StringLiteral",
                                "value": "\"Allocating GPU memory.\""
                            }
                        ],
                        "callee": {
                            "cform": "wbTime_start",
                            "loc": {
                                "end": {
                                    "column": 5,
                                    "line": 32
                                },
                                "start": {
                                    "column": 5,
                                    "line": 32
                                }
                            },
                            "name": "wbTime_start",
                            "raw": "wbTime_start(#kind, __VA_ARGS__)",
                            "type": "Identifier"
                        },
                        "cform": "wbTime_start(\"GPU\" /* String*/, \"Allocating GPU memory.\")",
                        "loc": {
                            "end": {
                                "column": 5,
                                "line": 32
                            },
                            "start": {
                                "column": 5,
                                "line": 32
                            }
                        },
                        "raw": "wbTime_start(#kind, __VA_ARGS__)",
                        "type": "CallExpression"
                    },
                    {},
                    {
                        "cform": "int  byteSize = sizeof(float ) * inputLength",
                        "declarations": [
                            {
                                "cform": "int  byteSize = sizeof(float ) * inputLength",
                                "id": {
                                    "cform": "byteSize",
                                    "loc": {
                                        "end": {
                                            "column": 35,
                                            "line": 34
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 34
                                        }
                                    },
                                    "name": "byteSize",
                                    "raw": "int byteSize =sizeof(float) * inputLength",
                                    "type": "Identifier"
                                },
                                "init": {
                                    "cform": "sizeof(float ) * inputLength",
                                    "left": {
                                        "arguments": [
                                            {
                                                "address_spaces": [],
                                                "bases": [
                                                    {
                                                        "cform": "float",
                                                        "loc": {
                                                            "end": {
                                                                "column": 31,
                                                                "line": 34
                                                            },
                                                            "start": {
                                                                "column": 19,
                                                                "line": 34
                                                            }
                                                        },
                                                        "raw": "float",
                                                        "type": "Literal",
                                                        "value": "float"
                                                    }
                                                ],
                                                "cform": "float ",
                                                "loc": {
                                                    "end": {
                                                        "column": 31,
                                                        "line": 34
                                                    },
                                                    "start": {
                                                        "column": 19,
                                                        "line": 34
                                                    }
                                                },
                                                "qualifiers": [],
                                                "raw": "sizeof(float)",
                                                "type": "TypeSpecification"
                                            }
                                        ],
                                        "callee": {
                                            "cform": "sizeof",
                                            "loc": {
                                                "end": {
                                                    "column": 31,
                                                    "line": 34
                                                },
                                                "start": {
                                                    "column": 19,
                                                    "line": 34
                                                }
                                            },
                                            "name": "sizeof",
                                            "raw": "sizeof(float)",
                                            "type": "Identifier"
                                        },
                                        "cform": "sizeof(float )",
                                        "loc": {
                                            "end": {
                                                "column": 31,
                                                "line": 34
                                            },
                                            "start": {
                                                "column": 19,
                                                "line": 34
                                            }
                                        },
                                        "raw": "sizeof(float)",
                                        "type": "CallExpression"
                                    },
                                    "loc": {
                                        "end": {
                                            "column": 35,
                                            "line": 34
                                        },
                                        "start": {
                                            "column": 19,
                                            "line": 34
                                        }
                                    },
                                    "operator": "*",
                                    "raw": "sizeof(float) * inputLength",
                                    "right": {
                                        "cform": "inputLength",
                                        "kind": {
                                            "address_spaces": [],
                                            "bases": [
                                                {
                                                    "cform": "int",
                                                    "loc": {
                                                        "end": {
                                                            "column": 9,
                                                            "line": 13
                                                        },
                                                        "start": {
                                                            "column": 5,
                                                            "line": 13
                                                        }
                                                    },
                                                    "raw": "int",
                                                    "type": "Literal",
                                                    "value": "int"
                                                }
                                            ],
                                            "cform": "int ",
                                            "loc": {
                                                "end": {
                                                    "column": 9,
                                                    "line": 13
                                                },
                                                "start": {
                                                    "column": 5,
                                                    "line": 13
                                                }
                                            },
                                            "qualifiers": [],
                                            "raw": "int inputLength",
                                            "type": "TypeSpecification"
                                        },
                                        "loc": {
                                            "end": {
                                                "column": 9,
                                                "line": 13
                                            },
                                            "start": {
                                                "column": 5,
                                                "line": 13
                                            }
                                        },
                                        "name": "inputLength",
                                        "raw": "int inputLength",
                                        "type": "Identifier"
                                    },
                                    "type": "BinaryExpression"
                                },
                                "kind": {
                                    "address_spaces": [],
                                    "bases": [
                                        {
                                            "cform": "int",
                                            "loc": {
                                                "end": {
                                                    "column": 35,
                                                    "line": 34
                                                },
                                                "start": {
                                                    "column": 5,
                                                    "line": 34
                                                }
                                            },
                                            "raw": "int",
                                            "type": "Literal",
                                            "value": "int"
                                        }
                                    ],
                                    "cform": "int ",
                                    "loc": {
                                        "end": {
                                            "column": 35,
                                            "line": 34
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 34
                                        }
                                    },
                                    "qualifiers": [],
                                    "raw": "int byteSize =sizeof(float) * inputLength",
                                    "type": "TypeSpecification"
                                },
                                "loc": {
                                    "end": {
                                        "column": 35,
                                        "line": 34
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 34
                                    }
                                },
                                "raw": "int byteSize =sizeof(float) * inputLength",
                                "type": "VariableDeclarator"
                            }
                        ],
                        "loc": {
                            "end": {
                                "column": 35,
                                "line": 34
                            },
                            "start": {
                                "column": 5,
                                "line": 34
                            }
                        },
                        "raw": "int byteSize =sizeof(float) * inputLength",
                        "type": "VariableDeclaration"
                    },
                    {
                        "arguments": [
                            {
                                "cform": "\"GPU\"",
                                "loc": {
                                    "end": {
                                        "column": 5,
                                        "line": 36
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 36
                                    }
                                },
                                "raw": "GPU",
                                "type": "StringLiteral",
                                "value": "\"GPU\""
                            },
                            {
                                "cform": "\"Allocating GPU memory.\"",
                                "loc": {
                                    "end": {
                                        "column": 5,
                                        "line": 36
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 36
                                    }
                                },
                                "raw": "Allocating GPU memory.",
                                "type": "StringLiteral",
                                "value": "\"Allocating GPU memory.\""
                            }
                        ],
                        "callee": {
                            "cform": "wbTime_stop",
                            "loc": {
                                "end": {
                                    "column": 5,
                                    "line": 36
                                },
                                "start": {
                                    "column": 5,
                                    "line": 36
                                }
                            },
                            "name": "wbTime_stop",
                            "raw": "wbTime_stop(#kind, __VA_ARGS__)",
                            "type": "Identifier"
                        },
                        "cform": "wbTime_stop(\"GPU\" /* String*/, \"Allocating GPU memory.\")",
                        "loc": {
                            "end": {
                                "column": 5,
                                "line": 36
                            },
                            "start": {
                                "column": 5,
                                "line": 36
                            }
                        },
                        "raw": "wbTime_stop(#kind, __VA_ARGS__)",
                        "type": "CallExpression"
                    },
                    {},
                    {
                        "arguments": [
                            {
                                "cform": "\"GPU\"",
                                "loc": {
                                    "end": {
                                        "column": 5,
                                        "line": 38
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 38
                                    }
                                },
                                "raw": "GPU",
                                "type": "StringLiteral",
                                "value": "\"GPU\""
                            },
                            {
                                "cform": "\"Copying input memory to the GPU.\"",
                                "loc": {
                                    "end": {
                                        "column": 5,
                                        "line": 38
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 38
                                    }
                                },
                                "raw": "Copying input memory to the GPU.",
                                "type": "StringLiteral",
                                "value": "\"Copying input memory to the GPU.\""
                            }
                        ],
                        "callee": {
                            "cform": "wbTime_start",
                            "loc": {
                                "end": {
                                    "column": 5,
                                    "line": 38
                                },
                                "start": {
                                    "column": 5,
                                    "line": 38
                                }
                            },
                            "name": "wbTime_start",
                            "raw": "wbTime_start(#kind, __VA_ARGS__)",
                            "type": "Identifier"
                        },
                        "cform": "wbTime_start(\"GPU\" /* String*/, \"Copying input memory to the GPU.\")",
                        "loc": {
                            "end": {
                                "column": 5,
                                "line": 38
                            },
                            "start": {
                                "column": 5,
                                "line": 38
                            }
                        },
                        "raw": "wbTime_start(#kind, __VA_ARGS__)",
                        "type": "CallExpression"
                    },
                    {},
                    {
                        "arguments": [
                            {
                                "argument": {
                                    "cform": "deviceInput1",
                                    "kind": {
                                        "cform": "float *",
                                        "loc": {
                                            "end": {
                                                "column": 13,
                                                "line": 17
                                            },
                                            "start": {
                                                "column": 5,
                                                "line": 17
                                            }
                                        },
                                        "raw": "float * deviceInput1",
                                        "type": "ReferenceType",
                                        "value": {
                                            "address_spaces": [],
                                            "bases": [
                                                {
                                                    "cform": "float",
                                                    "loc": {
                                                        "end": {
                                                            "column": 13,
                                                            "line": 17
                                                        },
                                                        "start": {
                                                            "column": 5,
                                                            "line": 17
                                                        }
                                                    },
                                                    "raw": "float",
                                                    "type": "Literal",
                                                    "value": "float"
                                                }
                                            ],
                                            "cform": "float ",
                                            "loc": {
                                                "end": {
                                                    "column": 13,
                                                    "line": 17
                                                },
                                                "start": {
                                                    "column": 5,
                                                    "line": 17
                                                }
                                            },
                                            "qualifiers": [],
                                            "raw": "float * deviceInput1",
                                            "type": "TypeSpecification"
                                        }
                                    },
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 17
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 17
                                        }
                                    },
                                    "name": "deviceInput1",
                                    "raw": "float * deviceInput1",
                                    "type": "Identifier"
                                },
                                "cform": "& deviceInput1",
                                "loc": {
                                    "end": {
                                        "column": 27,
                                        "line": 41
                                    },
                                    "start": {
                                        "column": 26,
                                        "line": 41
                                    }
                                },
                                "operator": "&",
                                "prefix": true,
                                "raw": "&deviceInput1",
                                "type": "UnaryExpression"
                            },
                            {
                                "cform": "byteSize",
                                "kind": {
                                    "address_spaces": [],
                                    "bases": [
                                        {
                                            "cform": "int",
                                            "loc": {
                                                "end": {
                                                    "column": 35,
                                                    "line": 34
                                                },
                                                "start": {
                                                    "column": 5,
                                                    "line": 34
                                                }
                                            },
                                            "raw": "int",
                                            "type": "Literal",
                                            "value": "int"
                                        }
                                    ],
                                    "cform": "int ",
                                    "loc": {
                                        "end": {
                                            "column": 35,
                                            "line": 34
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 34
                                        }
                                    },
                                    "qualifiers": [],
                                    "raw": "int byteSize =sizeof(float) * inputLength",
                                    "type": "TypeSpecification"
                                },
                                "loc": {
                                    "end": {
                                        "column": 35,
                                        "line": 34
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 34
                                    }
                                },
                                "name": "byteSize",
                                "raw": "int byteSize =sizeof(float) * inputLength",
                                "type": "Identifier"
                            }
                        ],
                        "callee": {
                            "cform": "cudaMalloc",
                            "loc": {
                                "end": {
                                    "column": 49,
                                    "line": 41
                                },
                                "start": {
                                    "column": 5,
                                    "line": 41
                                }
                            },
                            "name": "cudaMalloc",
                            "raw": "cudaMalloc((void **) &deviceInput1, byteSize)",
                            "type": "Identifier"
                        },
                        "cform": "cudaMalloc(& deviceInput1 /* UnaryOperator*/, byteSize)",
                        "loc": {
                            "end": {
                                "column": 49,
                                "line": 41
                            },
                            "start": {
                                "column": 5,
                                "line": 41
                            }
                        },
                        "raw": "cudaMalloc((void **) &deviceInput1, byteSize)",
                        "type": "CallExpression"
                    },
                    {
                        "arguments": [
                            {
                                "argument": {
                                    "cform": "deviceInput2",
                                    "kind": {
                                        "cform": "float *",
                                        "loc": {
                                            "end": {
                                                "column": 13,
                                                "line": 18
                                            },
                                            "start": {
                                                "column": 5,
                                                "line": 18
                                            }
                                        },
                                        "raw": "float * deviceInput2",
                                        "type": "ReferenceType",
                                        "value": {
                                            "address_spaces": [],
                                            "bases": [
                                                {
                                                    "cform": "float",
                                                    "loc": {
                                                        "end": {
                                                            "column": 13,
                                                            "line": 18
                                                        },
                                                        "start": {
                                                            "column": 5,
                                                            "line": 18
                                                        }
                                                    },
                                                    "raw": "float",
                                                    "type": "Literal",
                                                    "value": "float"
                                                }
                                            ],
                                            "cform": "float ",
                                            "loc": {
                                                "end": {
                                                    "column": 13,
                                                    "line": 18
                                                },
                                                "start": {
                                                    "column": 5,
                                                    "line": 18
                                                }
                                            },
                                            "qualifiers": [],
                                            "raw": "float * deviceInput2",
                                            "type": "TypeSpecification"
                                        }
                                    },
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 18
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 18
                                        }
                                    },
                                    "name": "deviceInput2",
                                    "raw": "float * deviceInput2",
                                    "type": "Identifier"
                                },
                                "cform": "& deviceInput2",
                                "loc": {
                                    "end": {
                                        "column": 27,
                                        "line": 42
                                    },
                                    "start": {
                                        "column": 26,
                                        "line": 42
                                    }
                                },
                                "operator": "&",
                                "prefix": true,
                                "raw": "&deviceInput2",
                                "type": "UnaryExpression"
                            },
                            {
                                "cform": "byteSize",
                                "kind": {
                                    "address_spaces": [],
                                    "bases": [
                                        {
                                            "cform": "int",
                                            "loc": {
                                                "end": {
                                                    "column": 35,
                                                    "line": 34
                                                },
                                                "start": {
                                                    "column": 5,
                                                    "line": 34
                                                }
                                            },
                                            "raw": "int",
                                            "type": "Literal",
                                            "value": "int"
                                        }
                                    ],
                                    "cform": "int ",
                                    "loc": {
                                        "end": {
                                            "column": 35,
                                            "line": 34
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 34
                                        }
                                    },
                                    "qualifiers": [],
                                    "raw": "int byteSize =sizeof(float) * inputLength",
                                    "type": "TypeSpecification"
                                },
                                "loc": {
                                    "end": {
                                        "column": 35,
                                        "line": 34
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 34
                                    }
                                },
                                "name": "byteSize",
                                "raw": "int byteSize =sizeof(float) * inputLength",
                                "type": "Identifier"
                            }
                        ],
                        "callee": {
                            "cform": "cudaMalloc",
                            "loc": {
                                "end": {
                                    "column": 49,
                                    "line": 42
                                },
                                "start": {
                                    "column": 5,
                                    "line": 42
                                }
                            },
                            "name": "cudaMalloc",
                            "raw": "cudaMalloc((void **) &deviceInput2, byteSize)",
                            "type": "Identifier"
                        },
                        "cform": "cudaMalloc(& deviceInput2 /* UnaryOperator*/, byteSize)",
                        "loc": {
                            "end": {
                                "column": 49,
                                "line": 42
                            },
                            "start": {
                                "column": 5,
                                "line": 42
                            }
                        },
                        "raw": "cudaMalloc((void **) &deviceInput2, byteSize)",
                        "type": "CallExpression"
                    },
                    {
                        "arguments": [
                            {
                                "argument": {
                                    "cform": "deviceOutput",
                                    "kind": {
                                        "cform": "float *",
                                        "loc": {
                                            "end": {
                                                "column": 13,
                                                "line": 19
                                            },
                                            "start": {
                                                "column": 5,
                                                "line": 19
                                            }
                                        },
                                        "raw": "float * deviceOutput",
                                        "type": "ReferenceType",
                                        "value": {
                                            "address_spaces": [],
                                            "bases": [
                                                {
                                                    "cform": "float",
                                                    "loc": {
                                                        "end": {
                                                            "column": 13,
                                                            "line": 19
                                                        },
                                                        "start": {
                                                            "column": 5,
                                                            "line": 19
                                                        }
                                                    },
                                                    "raw": "float",
                                                    "type": "Literal",
                                                    "value": "float"
                                                }
                                            ],
                                            "cform": "float ",
                                            "loc": {
                                                "end": {
                                                    "column": 13,
                                                    "line": 19
                                                },
                                                "start": {
                                                    "column": 5,
                                                    "line": 19
                                                }
                                            },
                                            "qualifiers": [],
                                            "raw": "float * deviceOutput",
                                            "type": "TypeSpecification"
                                        }
                                    },
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 19
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 19
                                        }
                                    },
                                    "name": "deviceOutput",
                                    "raw": "float * deviceOutput",
                                    "type": "Identifier"
                                },
                                "cform": "& deviceOutput",
                                "loc": {
                                    "end": {
                                        "column": 27,
                                        "line": 43
                                    },
                                    "start": {
                                        "column": 26,
                                        "line": 43
                                    }
                                },
                                "operator": "&",
                                "prefix": true,
                                "raw": "&deviceOutput",
                                "type": "UnaryExpression"
                            },
                            {
                                "cform": "byteSize",
                                "kind": {
                                    "address_spaces": [],
                                    "bases": [
                                        {
                                            "cform": "int",
                                            "loc": {
                                                "end": {
                                                    "column": 35,
                                                    "line": 34
                                                },
                                                "start": {
                                                    "column": 5,
                                                    "line": 34
                                                }
                                            },
                                            "raw": "int",
                                            "type": "Literal",
                                            "value": "int"
                                        }
                                    ],
                                    "cform": "int ",
                                    "loc": {
                                        "end": {
                                            "column": 35,
                                            "line": 34
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 34
                                        }
                                    },
                                    "qualifiers": [],
                                    "raw": "int byteSize =sizeof(float) * inputLength",
                                    "type": "TypeSpecification"
                                },
                                "loc": {
                                    "end": {
                                        "column": 35,
                                        "line": 34
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 34
                                    }
                                },
                                "name": "byteSize",
                                "raw": "int byteSize =sizeof(float) * inputLength",
                                "type": "Identifier"
                            }
                        ],
                        "callee": {
                            "cform": "cudaMalloc",
                            "loc": {
                                "end": {
                                    "column": 49,
                                    "line": 43
                                },
                                "start": {
                                    "column": 5,
                                    "line": 43
                                }
                            },
                            "name": "cudaMalloc",
                            "raw": "cudaMalloc((void **) &deviceOutput, byteSize)",
                            "type": "Identifier"
                        },
                        "cform": "cudaMalloc(& deviceOutput /* UnaryOperator*/, byteSize)",
                        "loc": {
                            "end": {
                                "column": 49,
                                "line": 43
                            },
                            "start": {
                                "column": 5,
                                "line": 43
                            }
                        },
                        "raw": "cudaMalloc((void **) &deviceOutput, byteSize)",
                        "type": "CallExpression"
                    },
                    {
                        "arguments": [
                            {
                                "cform": "\"GPU\"",
                                "loc": {
                                    "end": {
                                        "column": 5,
                                        "line": 46
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 46
                                    }
                                },
                                "raw": "GPU",
                                "type": "StringLiteral",
                                "value": "\"GPU\""
                            },
                            {
                                "cform": "\"Copying input memory to the GPU.\"",
                                "loc": {
                                    "end": {
                                        "column": 5,
                                        "line": 46
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 46
                                    }
                                },
                                "raw": "Copying input memory to the GPU.",
                                "type": "StringLiteral",
                                "value": "\"Copying input memory to the GPU.\""
                            }
                        ],
                        "callee": {
                            "cform": "wbTime_stop",
                            "loc": {
                                "end": {
                                    "column": 5,
                                    "line": 46
                                },
                                "start": {
                                    "column": 5,
                                    "line": 46
                                }
                            },
                            "name": "wbTime_stop",
                            "raw": "wbTime_stop(#kind, __VA_ARGS__)",
                            "type": "Identifier"
                        },
                        "cform": "wbTime_stop(\"GPU\" /* String*/, \"Copying input memory to the GPU.\")",
                        "loc": {
                            "end": {
                                "column": 5,
                                "line": 46
                            },
                            "start": {
                                "column": 5,
                                "line": 46
                            }
                        },
                        "raw": "wbTime_stop(#kind, __VA_ARGS__)",
                        "type": "CallExpression"
                    },
                    {},
                    {
                        "arguments": [
                            {
                                "cform": "deviceInput1",
                                "kind": {
                                    "cform": "float *",
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 17
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 17
                                        }
                                    },
                                    "raw": "float * deviceInput1",
                                    "type": "ReferenceType",
                                    "value": {
                                        "address_spaces": [],
                                        "bases": [
                                            {
                                                "cform": "float",
                                                "loc": {
                                                    "end": {
                                                        "column": 13,
                                                        "line": 17
                                                    },
                                                    "start": {
                                                        "column": 5,
                                                        "line": 17
                                                    }
                                                },
                                                "raw": "float",
                                                "type": "Literal",
                                                "value": "float"
                                            }
                                        ],
                                        "cform": "float ",
                                        "loc": {
                                            "end": {
                                                "column": 13,
                                                "line": 17
                                            },
                                            "start": {
                                                "column": 5,
                                                "line": 17
                                            }
                                        },
                                        "qualifiers": [],
                                        "raw": "float * deviceInput1",
                                        "type": "TypeSpecification"
                                    }
                                },
                                "loc": {
                                    "end": {
                                        "column": 13,
                                        "line": 17
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 17
                                    }
                                },
                                "name": "deviceInput1",
                                "raw": "float * deviceInput1",
                                "type": "Identifier"
                            },
                            {
                                "cform": "hostInput1",
                                "kind": {
                                    "cform": "float *",
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 14
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 14
                                        }
                                    },
                                    "raw": "float * hostInput1",
                                    "type": "ReferenceType",
                                    "value": {
                                        "address_spaces": [],
                                        "bases": [
                                            {
                                                "cform": "float",
                                                "loc": {
                                                    "end": {
                                                        "column": 13,
                                                        "line": 14
                                                    },
                                                    "start": {
                                                        "column": 5,
                                                        "line": 14
                                                    }
                                                },
                                                "raw": "float",
                                                "type": "Literal",
                                                "value": "float"
                                            }
                                        ],
                                        "cform": "float ",
                                        "loc": {
                                            "end": {
                                                "column": 13,
                                                "line": 14
                                            },
                                            "start": {
                                                "column": 5,
                                                "line": 14
                                            }
                                        },
                                        "qualifiers": [],
                                        "raw": "float * hostInput1",
                                        "type": "TypeSpecification"
                                    }
                                },
                                "loc": {
                                    "end": {
                                        "column": 13,
                                        "line": 14
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 14
                                    }
                                },
                                "name": "hostInput1",
                                "raw": "float * hostInput1",
                                "type": "Identifier"
                            },
                            {
                                "cform": "byteSize",
                                "kind": {
                                    "address_spaces": [],
                                    "bases": [
                                        {
                                            "cform": "int",
                                            "loc": {
                                                "end": {
                                                    "column": 35,
                                                    "line": 34
                                                },
                                                "start": {
                                                    "column": 5,
                                                    "line": 34
                                                }
                                            },
                                            "raw": "int",
                                            "type": "Literal",
                                            "value": "int"
                                        }
                                    ],
                                    "cform": "int ",
                                    "loc": {
                                        "end": {
                                            "column": 35,
                                            "line": 34
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 34
                                        }
                                    },
                                    "qualifiers": [],
                                    "raw": "int byteSize =sizeof(float) * inputLength",
                                    "type": "TypeSpecification"
                                },
                                "loc": {
                                    "end": {
                                        "column": 35,
                                        "line": 34
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 34
                                    }
                                },
                                "name": "byteSize",
                                "raw": "int byteSize =sizeof(float) * inputLength",
                                "type": "Identifier"
                            },
                            {
                                "cform": "cudaMemcpyHostToDevice",
                                "kind": {
                                    "address_spaces": [],
                                    "bases": [
                                        {
                                            "cform": "int",
                                            "loc": {
                                                "end": {
                                                    "column": 36,
                                                    "line": 40
                                                },
                                                "start": {
                                                    "column": 1,
                                                    "line": 40
                                                }
                                            },
                                            "raw": "int",
                                            "type": "Literal",
                                            "value": "int"
                                        }
                                    ],
                                    "cform": "const int ",
                                    "loc": {
                                        "end": {
                                            "column": 36,
                                            "line": 40
                                        },
                                        "start": {
                                            "column": 1,
                                            "line": 40
                                        }
                                    },
                                    "qualifiers": [
                                        {
                                            "cform": "const",
                                            "loc": {
                                                "end": {
                                                    "column": 36,
                                                    "line": 40
                                                },
                                                "start": {
                                                    "column": 1,
                                                    "line": 40
                                                }
                                            },
                                            "raw": "const int cudaMemcpyHostToDevice = 0",
                                            "type": "Literal",
                                            "value": "const"
                                        }
                                    ],
                                    "raw": "const int cudaMemcpyHostToDevice = 0",
                                    "type": "TypeSpecification"
                                },
                                "loc": {
                                    "end": {
                                        "column": 36,
                                        "line": 40
                                    },
                                    "start": {
                                        "column": 1,
                                        "line": 40
                                    }
                                },
                                "name": "cudaMemcpyHostToDevice",
                                "raw": "const int cudaMemcpyHostToDevice = 0",
                                "type": "Identifier"
                            }
                        ],
                        "callee": {
                            "cform": "cudaMemcpy",
                            "loc": {
                                "end": {
                                    "column": 73,
                                    "line": 49
                                },
                                "start": {
                                    "column": 5,
                                    "line": 49
                                }
                            },
                            "name": "cudaMemcpy",
                            "raw": "cudaMemcpy(deviceInput1, hostInput1, byteSize,cudaMemcpyHostToDevice)",
                            "type": "Identifier"
                        },
                        "cform": "cudaMemcpy(deviceInput1 /* Identifier*/, hostInput1 /* Identifier*/, byteSize /* Identifier*/, cudaMemcpyHostToDevice)",
                        "loc": {
                            "end": {
                                "column": 73,
                                "line": 49
                            },
                            "start": {
                                "column": 5,
                                "line": 49
                            }
                        },
                        "raw": "cudaMemcpy(deviceInput1, hostInput1, byteSize,cudaMemcpyHostToDevice)",
                        "type": "CallExpression"
                    },
                    {
                        "arguments": [
                            {
                                "cform": "deviceInput2",
                                "kind": {
                                    "cform": "float *",
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 18
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 18
                                        }
                                    },
                                    "raw": "float * deviceInput2",
                                    "type": "ReferenceType",
                                    "value": {
                                        "address_spaces": [],
                                        "bases": [
                                            {
                                                "cform": "float",
                                                "loc": {
                                                    "end": {
                                                        "column": 13,
                                                        "line": 18
                                                    },
                                                    "start": {
                                                        "column": 5,
                                                        "line": 18
                                                    }
                                                },
                                                "raw": "float",
                                                "type": "Literal",
                                                "value": "float"
                                            }
                                        ],
                                        "cform": "float ",
                                        "loc": {
                                            "end": {
                                                "column": 13,
                                                "line": 18
                                            },
                                            "start": {
                                                "column": 5,
                                                "line": 18
                                            }
                                        },
                                        "qualifiers": [],
                                        "raw": "float * deviceInput2",
                                        "type": "TypeSpecification"
                                    }
                                },
                                "loc": {
                                    "end": {
                                        "column": 13,
                                        "line": 18
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 18
                                    }
                                },
                                "name": "deviceInput2",
                                "raw": "float * deviceInput2",
                                "type": "Identifier"
                            },
                            {
                                "cform": "hostInput1",
                                "kind": {
                                    "cform": "float *",
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 14
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 14
                                        }
                                    },
                                    "raw": "float * hostInput1",
                                    "type": "ReferenceType",
                                    "value": {
                                        "address_spaces": [],
                                        "bases": [
                                            {
                                                "cform": "float",
                                                "loc": {
                                                    "end": {
                                                        "column": 13,
                                                        "line": 14
                                                    },
                                                    "start": {
                                                        "column": 5,
                                                        "line": 14
                                                    }
                                                },
                                                "raw": "float",
                                                "type": "Literal",
                                                "value": "float"
                                            }
                                        ],
                                        "cform": "float ",
                                        "loc": {
                                            "end": {
                                                "column": 13,
                                                "line": 14
                                            },
                                            "start": {
                                                "column": 5,
                                                "line": 14
                                            }
                                        },
                                        "qualifiers": [],
                                        "raw": "float * hostInput1",
                                        "type": "TypeSpecification"
                                    }
                                },
                                "loc": {
                                    "end": {
                                        "column": 13,
                                        "line": 14
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 14
                                    }
                                },
                                "name": "hostInput1",
                                "raw": "float * hostInput1",
                                "type": "Identifier"
                            },
                            {
                                "cform": "byteSize",
                                "kind": {
                                    "address_spaces": [],
                                    "bases": [
                                        {
                                            "cform": "int",
                                            "loc": {
                                                "end": {
                                                    "column": 35,
                                                    "line": 34
                                                },
                                                "start": {
                                                    "column": 5,
                                                    "line": 34
                                                }
                                            },
                                            "raw": "int",
                                            "type": "Literal",
                                            "value": "int"
                                        }
                                    ],
                                    "cform": "int ",
                                    "loc": {
                                        "end": {
                                            "column": 35,
                                            "line": 34
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 34
                                        }
                                    },
                                    "qualifiers": [],
                                    "raw": "int byteSize =sizeof(float) * inputLength",
                                    "type": "TypeSpecification"
                                },
                                "loc": {
                                    "end": {
                                        "column": 35,
                                        "line": 34
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 34
                                    }
                                },
                                "name": "byteSize",
                                "raw": "int byteSize =sizeof(float) * inputLength",
                                "type": "Identifier"
                            },
                            {
                                "cform": "cudaMemcpyHostToDevice",
                                "kind": {
                                    "address_spaces": [],
                                    "bases": [
                                        {
                                            "cform": "int",
                                            "loc": {
                                                "end": {
                                                    "column": 36,
                                                    "line": 40
                                                },
                                                "start": {
                                                    "column": 1,
                                                    "line": 40
                                                }
                                            },
                                            "raw": "int",
                                            "type": "Literal",
                                            "value": "int"
                                        }
                                    ],
                                    "cform": "const int ",
                                    "loc": {
                                        "end": {
                                            "column": 36,
                                            "line": 40
                                        },
                                        "start": {
                                            "column": 1,
                                            "line": 40
                                        }
                                    },
                                    "qualifiers": [
                                        {
                                            "cform": "const",
                                            "loc": {
                                                "end": {
                                                    "column": 36,
                                                    "line": 40
                                                },
                                                "start": {
                                                    "column": 1,
                                                    "line": 40
                                                }
                                            },
                                            "raw": "const int cudaMemcpyHostToDevice = 0",
                                            "type": "Literal",
                                            "value": "const"
                                        }
                                    ],
                                    "raw": "const int cudaMemcpyHostToDevice = 0",
                                    "type": "TypeSpecification"
                                },
                                "loc": {
                                    "end": {
                                        "column": 36,
                                        "line": 40
                                    },
                                    "start": {
                                        "column": 1,
                                        "line": 40
                                    }
                                },
                                "name": "cudaMemcpyHostToDevice",
                                "raw": "const int cudaMemcpyHostToDevice = 0",
                                "type": "Identifier"
                            }
                        ],
                        "callee": {
                            "cform": "cudaMemcpy",
                            "loc": {
                                "end": {
                                    "column": 73,
                                    "line": 51
                                },
                                "start": {
                                    "column": 5,
                                    "line": 51
                                }
                            },
                            "name": "cudaMemcpy",
                            "raw": "cudaMemcpy(deviceInput2, hostInput1, byteSize,cudaMemcpyHostToDevice)",
                            "type": "Identifier"
                        },
                        "cform": "cudaMemcpy(deviceInput2 /* Identifier*/, hostInput1 /* Identifier*/, byteSize /* Identifier*/, cudaMemcpyHostToDevice)",
                        "loc": {
                            "end": {
                                "column": 73,
                                "line": 51
                            },
                            "start": {
                                "column": 5,
                                "line": 51
                            }
                        },
                        "raw": "cudaMemcpy(deviceInput2, hostInput1, byteSize,cudaMemcpyHostToDevice)",
                        "type": "CallExpression"
                    },
                    {
                        "arguments": [
                            {
                                "cform": "\"Compute\"",
                                "loc": {
                                    "end": {
                                        "column": 5,
                                        "line": 54
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 54
                                    }
                                },
                                "raw": "Compute",
                                "type": "StringLiteral",
                                "value": "\"Compute\""
                            },
                            {
                                "cform": "\"Performing CUDA computation\"",
                                "loc": {
                                    "end": {
                                        "column": 5,
                                        "line": 54
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 54
                                    }
                                },
                                "raw": "Performing CUDA computation",
                                "type": "StringLiteral",
                                "value": "\"Performing CUDA computation\""
                            }
                        ],
                        "callee": {
                            "cform": "wbTime_start",
                            "loc": {
                                "end": {
                                    "column": 5,
                                    "line": 54
                                },
                                "start": {
                                    "column": 5,
                                    "line": 54
                                }
                            },
                            "name": "wbTime_start",
                            "raw": "wbTime_start(#kind, __VA_ARGS__)",
                            "type": "Identifier"
                        },
                        "cform": "wbTime_start(\"Compute\" /* String*/, \"Performing CUDA computation\")",
                        "loc": {
                            "end": {
                                "column": 5,
                                "line": 54
                            },
                            "start": {
                                "column": 5,
                                "line": 54
                            }
                        },
                        "raw": "wbTime_start(#kind, __VA_ARGS__)",
                        "type": "CallExpression"
                    },
                    {},
                    {
                        "cform": "int  block_size = 16",
                        "declarations": [
                            {
                                "cform": "int  block_size = 16",
                                "id": {
                                    "cform": "block_size",
                                    "loc": {
                                        "end": {
                                            "column": 23,
                                            "line": 56
                                        },
                                        "start": {
                                            "column": 6,
                                            "line": 56
                                        }
                                    },
                                    "name": "block_size",
                                    "raw": "int block_size = 16",
                                    "type": "Identifier"
                                },
                                "init": {
                                    "cform": "16",
                                    "loc": {
                                        "end": {
                                            "column": 23,
                                            "line": 56
                                        },
                                        "start": {
                                            "column": 23,
                                            "line": 56
                                        }
                                    },
                                    "raw": "16",
                                    "type": "Integer32Literal",
                                    "value": 16
                                },
                                "kind": {
                                    "address_spaces": [],
                                    "bases": [
                                        {
                                            "cform": "int",
                                            "loc": {
                                                "end": {
                                                    "column": 23,
                                                    "line": 56
                                                },
                                                "start": {
                                                    "column": 6,
                                                    "line": 56
                                                }
                                            },
                                            "raw": "int",
                                            "type": "Literal",
                                            "value": "int"
                                        }
                                    ],
                                    "cform": "int ",
                                    "loc": {
                                        "end": {
                                            "column": 23,
                                            "line": 56
                                        },
                                        "start": {
                                            "column": 6,
                                            "line": 56
                                        }
                                    },
                                    "qualifiers": [],
                                    "raw": "int block_size = 16",
                                    "type": "TypeSpecification"
                                },
                                "loc": {
                                    "end": {
                                        "column": 23,
                                        "line": 56
                                    },
                                    "start": {
                                        "column": 6,
                                        "line": 56
                                    }
                                },
                                "raw": "int block_size = 16",
                                "type": "VariableDeclarator"
                            }
                        ],
                        "loc": {
                            "end": {
                                "column": 23,
                                "line": 56
                            },
                            "start": {
                                "column": 6,
                                "line": 56
                            }
                        },
                        "raw": "int block_size = 16",
                        "type": "VariableDeclaration"
                    },
                    {
                        "cform": "int  n_blocks = inputLength / block_size + (inputLength % block_size == 0 ? 0 : 1)",
                        "declarations": [
                            {
                                "cform": "int  n_blocks = inputLength / block_size + (inputLength % block_size == 0 ? 0 : 1)",
                                "id": {
                                    "cform": "n_blocks",
                                    "loc": {
                                        "end": {
                                            "column": 81,
                                            "line": 57
                                        },
                                        "start": {
                                            "column": 6,
                                            "line": 57
                                        }
                                    },
                                    "name": "n_blocks",
                                    "raw": "int n_blocks = inputLength /block_size + (inputLength%block_size == 0 ? 0:1)",
                                    "type": "Identifier"
                                },
                                "init": {
                                    "cform": "inputLength / block_size + (inputLength % block_size == 0 ? 0 : 1)",
                                    "left": {
                                        "cform": "inputLength / block_size",
                                        "left": {
                                            "cform": "inputLength",
                                            "kind": {
                                                "address_spaces": [],
                                                "bases": [
                                                    {
                                                        "cform": "int",
                                                        "loc": {
                                                            "end": {
                                                                "column": 9,
                                                                "line": 13
                                                            },
                                                            "start": {
                                                                "column": 5,
                                                                "line": 13
                                                            }
                                                        },
                                                        "raw": "int",
                                                        "type": "Literal",
                                                        "value": "int"
                                                    }
                                                ],
                                                "cform": "int ",
                                                "loc": {
                                                    "end": {
                                                        "column": 9,
                                                        "line": 13
                                                    },
                                                    "start": {
                                                        "column": 5,
                                                        "line": 13
                                                    }
                                                },
                                                "qualifiers": [],
                                                "raw": "int inputLength",
                                                "type": "TypeSpecification"
                                            },
                                            "loc": {
                                                "end": {
                                                    "column": 9,
                                                    "line": 13
                                                },
                                                "start": {
                                                    "column": 5,
                                                    "line": 13
                                                }
                                            },
                                            "name": "inputLength",
                                            "raw": "int inputLength",
                                            "type": "Identifier"
                                        },
                                        "loc": {
                                            "end": {
                                                "column": 34,
                                                "line": 57
                                            },
                                            "start": {
                                                "column": 21,
                                                "line": 57
                                            }
                                        },
                                        "operator": "/",
                                        "raw": "inputLength /block_size",
                                        "right": {
                                            "cform": "block_size",
                                            "kind": {
                                                "address_spaces": [],
                                                "bases": [
                                                    {
                                                        "cform": "int",
                                                        "loc": {
                                                            "end": {
                                                                "column": 23,
                                                                "line": 56
                                                            },
                                                            "start": {
                                                                "column": 6,
                                                                "line": 56
                                                            }
                                                        },
                                                        "raw": "int",
                                                        "type": "Literal",
                                                        "value": "int"
                                                    }
                                                ],
                                                "cform": "int ",
                                                "loc": {
                                                    "end": {
                                                        "column": 23,
                                                        "line": 56
                                                    },
                                                    "start": {
                                                        "column": 6,
                                                        "line": 56
                                                    }
                                                },
                                                "qualifiers": [],
                                                "raw": "int block_size = 16",
                                                "type": "TypeSpecification"
                                            },
                                            "loc": {
                                                "end": {
                                                    "column": 23,
                                                    "line": 56
                                                },
                                                "start": {
                                                    "column": 6,
                                                    "line": 56
                                                }
                                            },
                                            "name": "block_size",
                                            "raw": "int block_size = 16",
                                            "type": "Identifier"
                                        },
                                        "type": "BinaryExpression"
                                    },
                                    "loc": {
                                        "end": {
                                            "column": 81,
                                            "line": 57
                                        },
                                        "start": {
                                            "column": 21,
                                            "line": 57
                                        }
                                    },
                                    "operator": "+",
                                    "raw": "inputLength /block_size + (inputLength%block_size == 0 ? 0:1)",
                                    "right": {
                                        "cform": "(inputLength % block_size == 0 ? 0 : 1)",
                                        "expression": {
                                            "alternate": {
                                                "cform": "1",
                                                "loc": {
                                                    "end": {
                                                        "column": 80,
                                                        "line": 57
                                                    },
                                                    "start": {
                                                        "column": 80,
                                                        "line": 57
                                                    }
                                                },
                                                "raw": "1",
                                                "type": "Integer32Literal",
                                                "value": 1
                                            },
                                            "cform": "inputLength % block_size == 0 ? 0 : 1",
                                            "consequent": {
                                                "cform": "0",
                                                "loc": {
                                                    "end": {
                                                        "column": 78,
                                                        "line": 57
                                                    },
                                                    "start": {
                                                        "column": 78,
                                                        "line": 57
                                                    }
                                                },
                                                "raw": "0",
                                                "type": "Integer32Literal",
                                                "value": 0
                                            },
                                            "loc": {
                                                "end": {
                                                    "column": 80,
                                                    "line": 57
                                                },
                                                "start": {
                                                    "column": 48,
                                                    "line": 57
                                                }
                                            },
                                            "raw": "inputLength%block_size == 0 ? 0:1",
                                            "test": {
                                                "cform": "inputLength % block_size == 0",
                                                "left": {
                                                    "cform": "inputLength % block_size",
                                                    "left": {
                                                        "cform": "inputLength",
                                                        "kind": {
                                                            "address_spaces": [],
                                                            "bases": [
                                                                {
                                                                    "cform": "int",
                                                                    "loc": {
                                                                        "end": {
                                                                            "column": 9,
                                                                            "line": 13
                                                                        },
                                                                        "start": {
                                                                            "column": 5,
                                                                            "line": 13
                                                                        }
                                                                    },
                                                                    "raw": "int",
                                                                    "type": "Literal",
                                                                    "value": "int"
                                                                }
                                                            ],
                                                            "cform": "int ",
                                                            "loc": {
                                                                "end": {
                                                                    "column": 9,
                                                                    "line": 13
                                                                },
                                                                "start": {
                                                                    "column": 5,
                                                                    "line": 13
                                                                }
                                                            },
                                                            "qualifiers": [],
                                                            "raw": "int inputLength",
                                                            "type": "TypeSpecification"
                                                        },
                                                        "loc": {
                                                            "end": {
                                                                "column": 9,
                                                                "line": 13
                                                            },
                                                            "start": {
                                                                "column": 5,
                                                                "line": 13
                                                            }
                                                        },
                                                        "name": "inputLength",
                                                        "raw": "int inputLength",
                                                        "type": "Identifier"
                                                    },
                                                    "loc": {
                                                        "end": {
                                                            "column": 60,
                                                            "line": 57
                                                        },
                                                        "start": {
                                                            "column": 48,
                                                            "line": 57
                                                        }
                                                    },
                                                    "operator": "%",
                                                    "raw": "inputLength%block_size",
                                                    "right": {
                                                        "cform": "block_size",
                                                        "kind": {
                                                            "address_spaces": [],
                                                            "bases": [
                                                                {
                                                                    "cform": "int",
                                                                    "loc": {
                                                                        "end": {
                                                                            "column": 23,
                                                                            "line": 56
                                                                        },
                                                                        "start": {
                                                                            "column": 6,
                                                                            "line": 56
                                                                        }
                                                                    },
                                                                    "raw": "int",
                                                                    "type": "Literal",
                                                                    "value": "int"
                                                                }
                                                            ],
                                                            "cform": "int ",
                                                            "loc": {
                                                                "end": {
                                                                    "column": 23,
                                                                    "line": 56
                                                                },
                                                                "start": {
                                                                    "column": 6,
                                                                    "line": 56
                                                                }
                                                            },
                                                            "qualifiers": [],
                                                            "raw": "int block_size = 16",
                                                            "type": "TypeSpecification"
                                                        },
                                                        "loc": {
                                                            "end": {
                                                                "column": 23,
                                                                "line": 56
                                                            },
                                                            "start": {
                                                                "column": 6,
                                                                "line": 56
                                                            }
                                                        },
                                                        "name": "block_size",
                                                        "raw": "int block_size = 16",
                                                        "type": "Identifier"
                                                    },
                                                    "type": "BinaryExpression"
                                                },
                                                "loc": {
                                                    "end": {
                                                        "column": 74,
                                                        "line": 57
                                                    },
                                                    "start": {
                                                        "column": 48,
                                                        "line": 57
                                                    }
                                                },
                                                "operator": "==",
                                                "raw": "inputLength%block_size == 0",
                                                "right": {
                                                    "cform": "0",
                                                    "loc": {
                                                        "end": {
                                                            "column": 74,
                                                            "line": 57
                                                        },
                                                        "start": {
                                                            "column": 74,
                                                            "line": 57
                                                        }
                                                    },
                                                    "raw": "0",
                                                    "type": "Integer32Literal",
                                                    "value": 0
                                                },
                                                "type": "BinaryExpression"
                                            },
                                            "type": "ConditionalExpression"
                                        },
                                        "loc": {
                                            "end": {
                                                "column": 81,
                                                "line": 57
                                            },
                                            "start": {
                                                "column": 47,
                                                "line": 57
                                            }
                                        },
                                        "raw": "(inputLength%block_size == 0 ? 0:1)",
                                        "type": "ExpressionStatement"
                                    },
                                    "type": "BinaryExpression"
                                },
                                "kind": {
                                    "address_spaces": [],
                                    "bases": [
                                        {
                                            "cform": "int",
                                            "loc": {
                                                "end": {
                                                    "column": 81,
                                                    "line": 57
                                                },
                                                "start": {
                                                    "column": 6,
                                                    "line": 57
                                                }
                                            },
                                            "raw": "int",
                                            "type": "Literal",
                                            "value": "int"
                                        }
                                    ],
                                    "cform": "int ",
                                    "loc": {
                                        "end": {
                                            "column": 81,
                                            "line": 57
                                        },
                                        "start": {
                                            "column": 6,
                                            "line": 57
                                        }
                                    },
                                    "qualifiers": [],
                                    "raw": "int n_blocks = inputLength /block_size + (inputLength%block_size == 0 ? 0:1)",
                                    "type": "TypeSpecification"
                                },
                                "loc": {
                                    "end": {
                                        "column": 81,
                                        "line": 57
                                    },
                                    "start": {
                                        "column": 6,
                                        "line": 57
                                    }
                                },
                                "raw": "int n_blocks = inputLength /block_size + (inputLength%block_size == 0 ? 0:1)",
                                "type": "VariableDeclarator"
                            }
                        ],
                        "loc": {
                            "end": {
                                "column": 81,
                                "line": 57
                            },
                            "start": {
                                "column": 6,
                                "line": 57
                            }
                        },
                        "raw": "int n_blocks = inputLength /block_size + (inputLength%block_size == 0 ? 0:1)",
                        "type": "VariableDeclaration"
                    },
                    {
                        "arguments": [
                            {
                                "cform": "deviceInput1",
                                "kind": {
                                    "cform": "float *",
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 17
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 17
                                        }
                                    },
                                    "raw": "float * deviceInput1",
                                    "type": "ReferenceType",
                                    "value": {
                                        "address_spaces": [],
                                        "bases": [
                                            {
                                                "cform": "float",
                                                "loc": {
                                                    "end": {
                                                        "column": 13,
                                                        "line": 17
                                                    },
                                                    "start": {
                                                        "column": 5,
                                                        "line": 17
                                                    }
                                                },
                                                "raw": "float",
                                                "type": "Literal",
                                                "value": "float"
                                            }
                                        ],
                                        "cform": "float ",
                                        "loc": {
                                            "end": {
                                                "column": 13,
                                                "line": 17
                                            },
                                            "start": {
                                                "column": 5,
                                                "line": 17
                                            }
                                        },
                                        "qualifiers": [],
                                        "raw": "float * deviceInput1",
                                        "type": "TypeSpecification"
                                    }
                                },
                                "loc": {
                                    "end": {
                                        "column": 13,
                                        "line": 17
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 17
                                    }
                                },
                                "name": "deviceInput1",
                                "raw": "float * deviceInput1",
                                "type": "Identifier"
                            },
                            {
                                "cform": "deviceInput2",
                                "kind": {
                                    "cform": "float *",
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 18
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 18
                                        }
                                    },
                                    "raw": "float * deviceInput2",
                                    "type": "ReferenceType",
                                    "value": {
                                        "address_spaces": [],
                                        "bases": [
                                            {
                                                "cform": "float",
                                                "loc": {
                                                    "end": {
                                                        "column": 13,
                                                        "line": 18
                                                    },
                                                    "start": {
                                                        "column": 5,
                                                        "line": 18
                                                    }
                                                },
                                                "raw": "float",
                                                "type": "Literal",
                                                "value": "float"
                                            }
                                        ],
                                        "cform": "float ",
                                        "loc": {
                                            "end": {
                                                "column": 13,
                                                "line": 18
                                            },
                                            "start": {
                                                "column": 5,
                                                "line": 18
                                            }
                                        },
                                        "qualifiers": [],
                                        "raw": "float * deviceInput2",
                                        "type": "TypeSpecification"
                                    }
                                },
                                "loc": {
                                    "end": {
                                        "column": 13,
                                        "line": 18
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 18
                                    }
                                },
                                "name": "deviceInput2",
                                "raw": "float * deviceInput2",
                                "type": "Identifier"
                            },
                            {
                                "cform": "deviceOutput",
                                "kind": {
                                    "cform": "float *",
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 19
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 19
                                        }
                                    },
                                    "raw": "float * deviceOutput",
                                    "type": "ReferenceType",
                                    "value": {
                                        "address_spaces": [],
                                        "bases": [
                                            {
                                                "cform": "float",
                                                "loc": {
                                                    "end": {
                                                        "column": 13,
                                                        "line": 19
                                                    },
                                                    "start": {
                                                        "column": 5,
                                                        "line": 19
                                                    }
                                                },
                                                "raw": "float",
                                                "type": "Literal",
                                                "value": "float"
                                            }
                                        ],
                                        "cform": "float ",
                                        "loc": {
                                            "end": {
                                                "column": 13,
                                                "line": 19
                                            },
                                            "start": {
                                                "column": 5,
                                                "line": 19
                                            }
                                        },
                                        "qualifiers": [],
                                        "raw": "float * deviceOutput",
                                        "type": "TypeSpecification"
                                    }
                                },
                                "loc": {
                                    "end": {
                                        "column": 13,
                                        "line": 19
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 19
                                    }
                                },
                                "name": "deviceOutput",
                                "raw": "float * deviceOutput",
                                "type": "Identifier"
                            },
                            {
                                "cform": "inputLength",
                                "kind": {
                                    "address_spaces": [],
                                    "bases": [
                                        {
                                            "cform": "int",
                                            "loc": {
                                                "end": {
                                                    "column": 9,
                                                    "line": 13
                                                },
                                                "start": {
                                                    "column": 5,
                                                    "line": 13
                                                }
                                            },
                                            "raw": "int",
                                            "type": "Literal",
                                            "value": "int"
                                        }
                                    ],
                                    "cform": "int ",
                                    "loc": {
                                        "end": {
                                            "column": 9,
                                            "line": 13
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 13
                                        }
                                    },
                                    "qualifiers": [],
                                    "raw": "int inputLength",
                                    "type": "TypeSpecification"
                                },
                                "loc": {
                                    "end": {
                                        "column": 9,
                                        "line": 13
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 13
                                    }
                                },
                                "name": "inputLength",
                                "raw": "int inputLength",
                                "type": "Identifier"
                            }
                        ],
                        "callee": "vecAdd",
                        "cform": "vecAdd<<<{n_blocks} /* CompoundNode*/, {block_size}>>>(deviceInput1 /* Identifier*/, deviceInput2 /* Identifier*/, deviceOutput /* Identifier*/, inputLength)",
                        "config": [
                            {
                                "cform": "{n_blocks}",
                                "elements": [
                                    {
                                        "cform": "n_blocks",
                                        "kind": {
                                            "address_spaces": [],
                                            "bases": [
                                                {
                                                    "cform": "int",
                                                    "loc": {
                                                        "end": {
                                                            "column": 81,
                                                            "line": 57
                                                        },
                                                        "start": {
                                                            "column": 6,
                                                            "line": 57
                                                        }
                                                    },
                                                    "raw": "int",
                                                    "type": "Literal",
                                                    "value": "int"
                                                }
                                            ],
                                            "cform": "int ",
                                            "loc": {
                                                "end": {
                                                    "column": 81,
                                                    "line": 57
                                                },
                                                "start": {
                                                    "column": 6,
                                                    "line": 57
                                                }
                                            },
                                            "qualifiers": [],
                                            "raw": "int n_blocks = inputLength /block_size + (inputLength%block_size == 0 ? 0:1)",
                                            "type": "TypeSpecification"
                                        },
                                        "loc": {
                                            "end": {
                                                "column": 81,
                                                "line": 57
                                            },
                                            "start": {
                                                "column": 6,
                                                "line": 57
                                            }
                                        },
                                        "name": "n_blocks",
                                        "raw": "int n_blocks = inputLength /block_size + (inputLength%block_size == 0 ? 0:1)",
                                        "type": "Identifier"
                                    }
                                ],
                                "loc": {
                                    "end": {
                                        "column": 15,
                                        "line": 60
                                    },
                                    "start": {
                                        "column": 15,
                                        "line": 60
                                    }
                                },
                                "raw": "n_blocks",
                                "type": "ArrayExpression"
                            },
                            {
                                "cform": "{block_size}",
                                "elements": [
                                    {
                                        "cform": "block_size",
                                        "kind": {
                                            "address_spaces": [],
                                            "bases": [
                                                {
                                                    "cform": "int",
                                                    "loc": {
                                                        "end": {
                                                            "column": 23,
                                                            "line": 56
                                                        },
                                                        "start": {
                                                            "column": 6,
                                                            "line": 56
                                                        }
                                                    },
                                                    "raw": "int",
                                                    "type": "Literal",
                                                    "value": "int"
                                                }
                                            ],
                                            "cform": "int ",
                                            "loc": {
                                                "end": {
                                                    "column": 23,
                                                    "line": 56
                                                },
                                                "start": {
                                                    "column": 6,
                                                    "line": 56
                                                }
                                            },
                                            "qualifiers": [],
                                            "raw": "int block_size = 16",
                                            "type": "TypeSpecification"
                                        },
                                        "loc": {
                                            "end": {
                                                "column": 23,
                                                "line": 56
                                            },
                                            "start": {
                                                "column": 6,
                                                "line": 56
                                            }
                                        },
                                        "name": "block_size",
                                        "raw": "int block_size = 16",
                                        "type": "Identifier"
                                    }
                                ],
                                "loc": {
                                    "end": {
                                        "column": 25,
                                        "line": 60
                                    },
                                    "start": {
                                        "column": 25,
                                        "line": 60
                                    }
                                },
                                "raw": "block_size",
                                "type": "ArrayExpression"
                            }
                        ],
                        "loc": {
                            "end": {
                                "column": 92,
                                "line": 60
                            },
                            "start": {
                                "column": 5,
                                "line": 60
                            }
                        },
                        "raw": "vecAdd<<< n_blocks, block_size>>>(deviceInput1, deviceInput2, deviceOutput, inputLength)",
                        "type": "CallExpression"
                    },
                    {
                        "arguments": [],
                        "callee": {
                            "cform": "cudaThreadSynchronize",
                            "loc": {
                                "end": {
                                    "column": 27,
                                    "line": 63
                                },
                                "start": {
                                    "column": 5,
                                    "line": 63
                                }
                            },
                            "name": "cudaThreadSynchronize",
                            "raw": "cudaThreadSynchronize()",
                            "type": "Identifier"
                        },
                        "cform": "cudaThreadSynchronize()",
                        "loc": {
                            "end": {
                                "column": 27,
                                "line": 63
                            },
                            "start": {
                                "column": 5,
                                "line": 63
                            }
                        },
                        "raw": "cudaThreadSynchronize()",
                        "type": "CallExpression"
                    },
                    {
                        "arguments": [
                            {
                                "cform": "\"Compute\"",
                                "loc": {
                                    "end": {
                                        "column": 5,
                                        "line": 64
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 64
                                    }
                                },
                                "raw": "Compute",
                                "type": "StringLiteral",
                                "value": "\"Compute\""
                            },
                            {
                                "cform": "\"Performing CUDA computation\"",
                                "loc": {
                                    "end": {
                                        "column": 5,
                                        "line": 64
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 64
                                    }
                                },
                                "raw": "Performing CUDA computation",
                                "type": "StringLiteral",
                                "value": "\"Performing CUDA computation\""
                            }
                        ],
                        "callee": {
                            "cform": "wbTime_stop",
                            "loc": {
                                "end": {
                                    "column": 5,
                                    "line": 64
                                },
                                "start": {
                                    "column": 5,
                                    "line": 64
                                }
                            },
                            "name": "wbTime_stop",
                            "raw": "wbTime_stop(#kind, __VA_ARGS__)",
                            "type": "Identifier"
                        },
                        "cform": "wbTime_stop(\"Compute\" /* String*/, \"Performing CUDA computation\")",
                        "loc": {
                            "end": {
                                "column": 5,
                                "line": 64
                            },
                            "start": {
                                "column": 5,
                                "line": 64
                            }
                        },
                        "raw": "wbTime_stop(#kind, __VA_ARGS__)",
                        "type": "CallExpression"
                    },
                    {},
                    {
                        "arguments": [
                            {
                                "cform": "\"Copy\"",
                                "loc": {
                                    "end": {
                                        "column": 5,
                                        "line": 66
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 66
                                    }
                                },
                                "raw": "Copy",
                                "type": "StringLiteral",
                                "value": "\"Copy\""
                            },
                            {
                                "cform": "\"Copying output memory to the CPU\"",
                                "loc": {
                                    "end": {
                                        "column": 5,
                                        "line": 66
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 66
                                    }
                                },
                                "raw": "Copying output memory to the CPU",
                                "type": "StringLiteral",
                                "value": "\"Copying output memory to the CPU\""
                            }
                        ],
                        "callee": {
                            "cform": "wbTime_start",
                            "loc": {
                                "end": {
                                    "column": 5,
                                    "line": 66
                                },
                                "start": {
                                    "column": 5,
                                    "line": 66
                                }
                            },
                            "name": "wbTime_start",
                            "raw": "wbTime_start(#kind, __VA_ARGS__)",
                            "type": "Identifier"
                        },
                        "cform": "wbTime_start(\"Copy\" /* String*/, \"Copying output memory to the CPU\")",
                        "loc": {
                            "end": {
                                "column": 5,
                                "line": 66
                            },
                            "start": {
                                "column": 5,
                                "line": 66
                            }
                        },
                        "raw": "wbTime_start(#kind, __VA_ARGS__)",
                        "type": "CallExpression"
                    },
                    {},
                    {
                        "arguments": [
                            {
                                "cform": "hostOutput",
                                "kind": {
                                    "cform": "float *",
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 16
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 16
                                        }
                                    },
                                    "raw": "float * hostOutput",
                                    "type": "ReferenceType",
                                    "value": {
                                        "address_spaces": [],
                                        "bases": [
                                            {
                                                "cform": "float",
                                                "loc": {
                                                    "end": {
                                                        "column": 13,
                                                        "line": 16
                                                    },
                                                    "start": {
                                                        "column": 5,
                                                        "line": 16
                                                    }
                                                },
                                                "raw": "float",
                                                "type": "Literal",
                                                "value": "float"
                                            }
                                        ],
                                        "cform": "float ",
                                        "loc": {
                                            "end": {
                                                "column": 13,
                                                "line": 16
                                            },
                                            "start": {
                                                "column": 5,
                                                "line": 16
                                            }
                                        },
                                        "qualifiers": [],
                                        "raw": "float * hostOutput",
                                        "type": "TypeSpecification"
                                    }
                                },
                                "loc": {
                                    "end": {
                                        "column": 13,
                                        "line": 16
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 16
                                    }
                                },
                                "name": "hostOutput",
                                "raw": "float * hostOutput",
                                "type": "Identifier"
                            },
                            {
                                "cform": "deviceOutput",
                                "kind": {
                                    "cform": "float *",
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 19
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 19
                                        }
                                    },
                                    "raw": "float * deviceOutput",
                                    "type": "ReferenceType",
                                    "value": {
                                        "address_spaces": [],
                                        "bases": [
                                            {
                                                "cform": "float",
                                                "loc": {
                                                    "end": {
                                                        "column": 13,
                                                        "line": 19
                                                    },
                                                    "start": {
                                                        "column": 5,
                                                        "line": 19
                                                    }
                                                },
                                                "raw": "float",
                                                "type": "Literal",
                                                "value": "float"
                                            }
                                        ],
                                        "cform": "float ",
                                        "loc": {
                                            "end": {
                                                "column": 13,
                                                "line": 19
                                            },
                                            "start": {
                                                "column": 5,
                                                "line": 19
                                            }
                                        },
                                        "qualifiers": [],
                                        "raw": "float * deviceOutput",
                                        "type": "TypeSpecification"
                                    }
                                },
                                "loc": {
                                    "end": {
                                        "column": 13,
                                        "line": 19
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 19
                                    }
                                },
                                "name": "deviceOutput",
                                "raw": "float * deviceOutput",
                                "type": "Identifier"
                            },
                            {
                                "cform": "byteSize",
                                "kind": {
                                    "address_spaces": [],
                                    "bases": [
                                        {
                                            "cform": "int",
                                            "loc": {
                                                "end": {
                                                    "column": 35,
                                                    "line": 34
                                                },
                                                "start": {
                                                    "column": 5,
                                                    "line": 34
                                                }
                                            },
                                            "raw": "int",
                                            "type": "Literal",
                                            "value": "int"
                                        }
                                    ],
                                    "cform": "int ",
                                    "loc": {
                                        "end": {
                                            "column": 35,
                                            "line": 34
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 34
                                        }
                                    },
                                    "qualifiers": [],
                                    "raw": "int byteSize =sizeof(float) * inputLength",
                                    "type": "TypeSpecification"
                                },
                                "loc": {
                                    "end": {
                                        "column": 35,
                                        "line": 34
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 34
                                    }
                                },
                                "name": "byteSize",
                                "raw": "int byteSize =sizeof(float) * inputLength",
                                "type": "Identifier"
                            },
                            {
                                "cform": "cudaMemcpyDeviceToHost",
                                "kind": {
                                    "address_spaces": [],
                                    "bases": [
                                        {
                                            "cform": "int",
                                            "loc": {
                                                "end": {
                                                    "column": 36,
                                                    "line": 41
                                                },
                                                "start": {
                                                    "column": 1,
                                                    "line": 41
                                                }
                                            },
                                            "raw": "int",
                                            "type": "Literal",
                                            "value": "int"
                                        }
                                    ],
                                    "cform": "const int ",
                                    "loc": {
                                        "end": {
                                            "column": 36,
                                            "line": 41
                                        },
                                        "start": {
                                            "column": 1,
                                            "line": 41
                                        }
                                    },
                                    "qualifiers": [
                                        {
                                            "cform": "const",
                                            "loc": {
                                                "end": {
                                                    "column": 36,
                                                    "line": 41
                                                },
                                                "start": {
                                                    "column": 1,
                                                    "line": 41
                                                }
                                            },
                                            "raw": "const int cudaMemcpyDeviceToHost = 1",
                                            "type": "Literal",
                                            "value": "const"
                                        }
                                    ],
                                    "raw": "const int cudaMemcpyDeviceToHost = 1",
                                    "type": "TypeSpecification"
                                },
                                "loc": {
                                    "end": {
                                        "column": 36,
                                        "line": 41
                                    },
                                    "start": {
                                        "column": 1,
                                        "line": 41
                                    }
                                },
                                "name": "cudaMemcpyDeviceToHost",
                                "raw": "const int cudaMemcpyDeviceToHost = 1",
                                "type": "Identifier"
                            }
                        ],
                        "callee": {
                            "cform": "cudaMemcpy",
                            "loc": {
                                "end": {
                                    "column": 73,
                                    "line": 68
                                },
                                "start": {
                                    "column": 5,
                                    "line": 68
                                }
                            },
                            "name": "cudaMemcpy",
                            "raw": "cudaMemcpy(hostOutput, deviceOutput, byteSize,cudaMemcpyDeviceToHost)",
                            "type": "Identifier"
                        },
                        "cform": "cudaMemcpy(hostOutput /* Identifier*/, deviceOutput /* Identifier*/, byteSize /* Identifier*/, cudaMemcpyDeviceToHost)",
                        "loc": {
                            "end": {
                                "column": 73,
                                "line": 68
                            },
                            "start": {
                                "column": 5,
                                "line": 68
                            }
                        },
                        "raw": "cudaMemcpy(hostOutput, deviceOutput, byteSize,cudaMemcpyDeviceToHost)",
                        "type": "CallExpression"
                    },
                    {
                        "arguments": [
                            {
                                "cform": "\"Copy\"",
                                "loc": {
                                    "end": {
                                        "column": 5,
                                        "line": 70
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 70
                                    }
                                },
                                "raw": "Copy",
                                "type": "StringLiteral",
                                "value": "\"Copy\""
                            },
                            {
                                "cform": "\"Copying output memory to the CPU\"",
                                "loc": {
                                    "end": {
                                        "column": 5,
                                        "line": 70
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 70
                                    }
                                },
                                "raw": "Copying output memory to the CPU",
                                "type": "StringLiteral",
                                "value": "\"Copying output memory to the CPU\""
                            }
                        ],
                        "callee": {
                            "cform": "wbTime_stop",
                            "loc": {
                                "end": {
                                    "column": 5,
                                    "line": 70
                                },
                                "start": {
                                    "column": 5,
                                    "line": 70
                                }
                            },
                            "name": "wbTime_stop",
                            "raw": "wbTime_stop(#kind, __VA_ARGS__)",
                            "type": "Identifier"
                        },
                        "cform": "wbTime_stop(\"Copy\" /* String*/, \"Copying output memory to the CPU\")",
                        "loc": {
                            "end": {
                                "column": 5,
                                "line": 70
                            },
                            "start": {
                                "column": 5,
                                "line": 70
                            }
                        },
                        "raw": "wbTime_stop(#kind, __VA_ARGS__)",
                        "type": "CallExpression"
                    },
                    {},
                    {
                        "arguments": [
                            {
                                "cform": "\"GPU\"",
                                "loc": {
                                    "end": {
                                        "column": 5,
                                        "line": 72
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 72
                                    }
                                },
                                "raw": "GPU",
                                "type": "StringLiteral",
                                "value": "\"GPU\""
                            },
                            {
                                "cform": "\"Freeing GPU Memory\"",
                                "loc": {
                                    "end": {
                                        "column": 5,
                                        "line": 72
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 72
                                    }
                                },
                                "raw": "Freeing GPU Memory",
                                "type": "StringLiteral",
                                "value": "\"Freeing GPU Memory\""
                            }
                        ],
                        "callee": {
                            "cform": "wbTime_start",
                            "loc": {
                                "end": {
                                    "column": 5,
                                    "line": 72
                                },
                                "start": {
                                    "column": 5,
                                    "line": 72
                                }
                            },
                            "name": "wbTime_start",
                            "raw": "wbTime_start(#kind, __VA_ARGS__)",
                            "type": "Identifier"
                        },
                        "cform": "wbTime_start(\"GPU\" /* String*/, \"Freeing GPU Memory\")",
                        "loc": {
                            "end": {
                                "column": 5,
                                "line": 72
                            },
                            "start": {
                                "column": 5,
                                "line": 72
                            }
                        },
                        "raw": "wbTime_start(#kind, __VA_ARGS__)",
                        "type": "CallExpression"
                    },
                    {},
                    {
                        "arguments": [
                            {
                                "cform": "\"GPU\"",
                                "loc": {
                                    "end": {
                                        "column": 5,
                                        "line": 76
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 76
                                    }
                                },
                                "raw": "GPU",
                                "type": "StringLiteral",
                                "value": "\"GPU\""
                            },
                            {
                                "cform": "\"Freeing GPU Memory\"",
                                "loc": {
                                    "end": {
                                        "column": 5,
                                        "line": 76
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 76
                                    }
                                },
                                "raw": "Freeing GPU Memory",
                                "type": "StringLiteral",
                                "value": "\"Freeing GPU Memory\""
                            }
                        ],
                        "callee": {
                            "cform": "wbTime_stop",
                            "loc": {
                                "end": {
                                    "column": 5,
                                    "line": 76
                                },
                                "start": {
                                    "column": 5,
                                    "line": 76
                                }
                            },
                            "name": "wbTime_stop",
                            "raw": "wbTime_stop(#kind, __VA_ARGS__)",
                            "type": "Identifier"
                        },
                        "cform": "wbTime_stop(\"GPU\" /* String*/, \"Freeing GPU Memory\")",
                        "loc": {
                            "end": {
                                "column": 5,
                                "line": 76
                            },
                            "start": {
                                "column": 5,
                                "line": 76
                            }
                        },
                        "raw": "wbTime_stop(#kind, __VA_ARGS__)",
                        "type": "CallExpression"
                    },
                    {},
                    {
                        "arguments": [
                            {
                                "cform": "args",
                                "kind": {
                                    "address_spaces": [],
                                    "bases": [
                                        {
                                            "cform": "int",
                                            "loc": {
                                                "end": {
                                                    "column": 13,
                                                    "line": 12
                                                },
                                                "start": {
                                                    "column": 5,
                                                    "line": 12
                                                }
                                            },
                                            "raw": "int",
                                            "type": "Literal",
                                            "value": "int"
                                        }
                                    ],
                                    "cform": "int ",
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 12
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 12
                                        }
                                    },
                                    "qualifiers": [],
                                    "raw": "wbArg_t args",
                                    "type": "TypeSpecification"
                                },
                                "loc": {
                                    "end": {
                                        "column": 13,
                                        "line": 12
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 12
                                    }
                                },
                                "name": "args",
                                "raw": "wbArg_t args",
                                "type": "Identifier"
                            },
                            {
                                "cform": "hostOutput",
                                "kind": {
                                    "cform": "float *",
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 16
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 16
                                        }
                                    },
                                    "raw": "float * hostOutput",
                                    "type": "ReferenceType",
                                    "value": {
                                        "address_spaces": [],
                                        "bases": [
                                            {
                                                "cform": "float",
                                                "loc": {
                                                    "end": {
                                                        "column": 13,
                                                        "line": 16
                                                    },
                                                    "start": {
                                                        "column": 5,
                                                        "line": 16
                                                    }
                                                },
                                                "raw": "float",
                                                "type": "Literal",
                                                "value": "float"
                                            }
                                        ],
                                        "cform": "float ",
                                        "loc": {
                                            "end": {
                                                "column": 13,
                                                "line": 16
                                            },
                                            "start": {
                                                "column": 5,
                                                "line": 16
                                            }
                                        },
                                        "qualifiers": [],
                                        "raw": "float * hostOutput",
                                        "type": "TypeSpecification"
                                    }
                                },
                                "loc": {
                                    "end": {
                                        "column": 13,
                                        "line": 16
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 16
                                    }
                                },
                                "name": "hostOutput",
                                "raw": "float * hostOutput",
                                "type": "Identifier"
                            },
                            {
                                "cform": "inputLength",
                                "kind": {
                                    "address_spaces": [],
                                    "bases": [
                                        {
                                            "cform": "int",
                                            "loc": {
                                                "end": {
                                                    "column": 9,
                                                    "line": 13
                                                },
                                                "start": {
                                                    "column": 5,
                                                    "line": 13
                                                }
                                            },
                                            "raw": "int",
                                            "type": "Literal",
                                            "value": "int"
                                        }
                                    ],
                                    "cform": "int ",
                                    "loc": {
                                        "end": {
                                            "column": 9,
                                            "line": 13
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 13
                                        }
                                    },
                                    "qualifiers": [],
                                    "raw": "int inputLength",
                                    "type": "TypeSpecification"
                                },
                                "loc": {
                                    "end": {
                                        "column": 9,
                                        "line": 13
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 13
                                    }
                                },
                                "name": "inputLength",
                                "raw": "int inputLength",
                                "type": "Identifier"
                            }
                        ],
                        "callee": {
                            "cform": "wbSolution",
                            "loc": {
                                "end": {
                                    "column": 45,
                                    "line": 78
                                },
                                "start": {
                                    "column": 5,
                                    "line": 78
                                }
                            },
                            "name": "wbSolution",
                            "raw": "wbSolution(args, hostOutput, inputLength)",
                            "type": "Identifier"
                        },
                        "cform": "wbSolution(args /* Identifier*/, hostOutput /* Identifier*/, inputLength)",
                        "loc": {
                            "end": {
                                "column": 45,
                                "line": 78
                            },
                            "start": {
                                "column": 5,
                                "line": 78
                            }
                        },
                        "raw": "wbSolution(args, hostOutput, inputLength)",
                        "type": "CallExpression"
                    },
                    {
                        "arguments": [
                            {
                                "cform": "hostInput1",
                                "kind": {
                                    "cform": "float *",
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 14
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 14
                                        }
                                    },
                                    "raw": "float * hostInput1",
                                    "type": "ReferenceType",
                                    "value": {
                                        "address_spaces": [],
                                        "bases": [
                                            {
                                                "cform": "float",
                                                "loc": {
                                                    "end": {
                                                        "column": 13,
                                                        "line": 14
                                                    },
                                                    "start": {
                                                        "column": 5,
                                                        "line": 14
                                                    }
                                                },
                                                "raw": "float",
                                                "type": "Literal",
                                                "value": "float"
                                            }
                                        ],
                                        "cform": "float ",
                                        "loc": {
                                            "end": {
                                                "column": 13,
                                                "line": 14
                                            },
                                            "start": {
                                                "column": 5,
                                                "line": 14
                                            }
                                        },
                                        "qualifiers": [],
                                        "raw": "float * hostInput1",
                                        "type": "TypeSpecification"
                                    }
                                },
                                "loc": {
                                    "end": {
                                        "column": 13,
                                        "line": 14
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 14
                                    }
                                },
                                "name": "hostInput1",
                                "raw": "float * hostInput1",
                                "type": "Identifier"
                            }
                        ],
                        "callee": {
                            "cform": "free",
                            "loc": {
                                "end": {
                                    "column": 20,
                                    "line": 80
                                },
                                "start": {
                                    "column": 5,
                                    "line": 80
                                }
                            },
                            "name": "free",
                            "raw": "free(hostInput1)",
                            "type": "Identifier"
                        },
                        "cform": "free(hostInput1)",
                        "loc": {
                            "end": {
                                "column": 20,
                                "line": 80
                            },
                            "start": {
                                "column": 5,
                                "line": 80
                            }
                        },
                        "raw": "free(hostInput1)",
                        "type": "CallExpression"
                    },
                    {
                        "arguments": [
                            {
                                "cform": "hostInput2",
                                "kind": {
                                    "cform": "float *",
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 15
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 15
                                        }
                                    },
                                    "raw": "float * hostInput2",
                                    "type": "ReferenceType",
                                    "value": {
                                        "address_spaces": [],
                                        "bases": [
                                            {
                                                "cform": "float",
                                                "loc": {
                                                    "end": {
                                                        "column": 13,
                                                        "line": 15
                                                    },
                                                    "start": {
                                                        "column": 5,
                                                        "line": 15
                                                    }
                                                },
                                                "raw": "float",
                                                "type": "Literal",
                                                "value": "float"
                                            }
                                        ],
                                        "cform": "float ",
                                        "loc": {
                                            "end": {
                                                "column": 13,
                                                "line": 15
                                            },
                                            "start": {
                                                "column": 5,
                                                "line": 15
                                            }
                                        },
                                        "qualifiers": [],
                                        "raw": "float * hostInput2",
                                        "type": "TypeSpecification"
                                    }
                                },
                                "loc": {
                                    "end": {
                                        "column": 13,
                                        "line": 15
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 15
                                    }
                                },
                                "name": "hostInput2",
                                "raw": "float * hostInput2",
                                "type": "Identifier"
                            }
                        ],
                        "callee": {
                            "cform": "free",
                            "loc": {
                                "end": {
                                    "column": 20,
                                    "line": 81
                                },
                                "start": {
                                    "column": 5,
                                    "line": 81
                                }
                            },
                            "name": "free",
                            "raw": "free(hostInput2)",
                            "type": "Identifier"
                        },
                        "cform": "free(hostInput2)",
                        "loc": {
                            "end": {
                                "column": 20,
                                "line": 81
                            },
                            "start": {
                                "column": 5,
                                "line": 81
                            }
                        },
                        "raw": "free(hostInput2)",
                        "type": "CallExpression"
                    },
                    {
                        "arguments": [
                            {
                                "cform": "hostOutput",
                                "kind": {
                                    "cform": "float *",
                                    "loc": {
                                        "end": {
                                            "column": 13,
                                            "line": 16
                                        },
                                        "start": {
                                            "column": 5,
                                            "line": 16
                                        }
                                    },
                                    "raw": "float * hostOutput",
                                    "type": "ReferenceType",
                                    "value": {
                                        "address_spaces": [],
                                        "bases": [
                                            {
                                                "cform": "float",
                                                "loc": {
                                                    "end": {
                                                        "column": 13,
                                                        "line": 16
                                                    },
                                                    "start": {
                                                        "column": 5,
                                                        "line": 16
                                                    }
                                                },
                                                "raw": "float",
                                                "type": "Literal",
                                                "value": "float"
                                            }
                                        ],
                                        "cform": "float ",
                                        "loc": {
                                            "end": {
                                                "column": 13,
                                                "line": 16
                                            },
                                            "start": {
                                                "column": 5,
                                                "line": 16
                                            }
                                        },
                                        "qualifiers": [],
                                        "raw": "float * hostOutput",
                                        "type": "TypeSpecification"
                                    }
                                },
                                "loc": {
                                    "end": {
                                        "column": 13,
                                        "line": 16
                                    },
                                    "start": {
                                        "column": 5,
                                        "line": 16
                                    }
                                },
                                "name": "hostOutput",
                                "raw": "float * hostOutput",
                                "type": "Identifier"
                            }
                        ],
                        "callee": {
                            "cform": "free",
                            "loc": {
                                "end": {
                                    "column": 20,
                                    "line": 82
                                },
                                "start": {
                                    "column": 5,
                                    "line": 82
                                }
                            },
                            "name": "free",
                            "raw": "free(hostOutput)",
                            "type": "Identifier"
                        },
                        "cform": "free(hostOutput)",
                        "loc": {
                            "end": {
                                "column": 20,
                                "line": 82
                            },
                            "start": {
                                "column": 5,
                                "line": 82
                            }
                        },
                        "raw": "free(hostOutput)",
                        "type": "CallExpression"
                    },
                    {
                        "argument": {
                            "cform": "0",
                            "loc": {
                                "end": {
                                    "column": 12,
                                    "line": 84
                                },
                                "start": {
                                    "column": 12,
                                    "line": 84
                                }
                            },
                            "raw": "0",
                            "type": "Integer32Literal",
                            "value": 0
                        },
                        "cform": "return 0",
                        "loc": {
                            "end": {
                                "column": 12,
                                "line": 84
                            },
                            "start": {
                                "column": 5,
                                "line": 84
                            }
                        },
                        "raw": "return 0",
                        "type": "ReturnStatement"
                    }
                ],
                "cform": "{\nint  args; /* Declare*/\nint  inputLength; /* Declare*/\nfloat * hostInput1; /* Declare*/\nfloat * hostInput2; /* Declare*/\nfloat * hostOutput; /* Declare*/\nfloat * deviceInput1; /* Declare*/\nfloat * deviceInput2; /* Declare*/\nfloat * deviceOutput; /* Declare*/\nargs = wbArg_read(argc /* Identifier*/, argv); /* Assign*/\nwbTime_start(\"Generic\" /* String*/, \"Importing data and creating memory on host\"); /* Call*/\nhostInput1 = wbImport(\"input0\" /* String*/, & inputLength); /* Assign*/\nhostInput2 = wbImport(\"input1\" /* String*/, & inputLength); /* Assign*/\nhostOutput = malloc(inputLength * sizeof(float )); /* Assign*/\nwbTime_stop(\"Generic\" /* String*/, \"Importing data and creating memory on host\"); /* Call*/\nwbLog(\"TRACE\" /* String*/, \"The input length is \" /* String*/, inputLength /* Identifier*/, \" elements\"); /* Call*/\nwbTime_start(\"GPU\" /* String*/, \"Allocating GPU memory.\"); /* Call*/\nint  byteSize = sizeof(float ) * inputLength; /* Declare*/\nwbTime_stop(\"GPU\" /* String*/, \"Allocating GPU memory.\"); /* Call*/\nwbTime_start(\"GPU\" /* String*/, \"Copying input memory to the GPU.\"); /* Call*/\ncudaMalloc(& deviceInput1 /* UnaryOperator*/, byteSize); /* Call*/\ncudaMalloc(& deviceInput2 /* UnaryOperator*/, byteSize); /* Call*/\ncudaMalloc(& deviceOutput /* UnaryOperator*/, byteSize); /* Call*/\nwbTime_stop(\"GPU\" /* String*/, \"Copying input memory to the GPU.\"); /* Call*/\ncudaMemcpy(deviceInput1 /* Identifier*/, hostInput1 /* Identifier*/, byteSize /* Identifier*/, cudaMemcpyHostToDevice); /* Call*/\ncudaMemcpy(deviceInput2 /* Identifier*/, hostInput1 /* Identifier*/, byteSize /* Identifier*/, cudaMemcpyHostToDevice); /* Call*/\nwbTime_start(\"Compute\" /* String*/, \"Performing CUDA computation\"); /* Call*/\nint  block_size = 16; /* Declare*/\nint  n_blocks = inputLength / block_size + (inputLength % block_size == 0 ? 0 : 1); /* Declare*/\nvecAdd<<<{n_blocks} /* CompoundNode*/, {block_size}>>>(deviceInput1 /* Identifier*/, deviceInput2 /* Identifier*/, deviceOutput /* Identifier*/, inputLength); /* Call*/\ncudaThreadSynchronize(); /* Call*/\nwbTime_stop(\"Compute\" /* String*/, \"Performing CUDA computation\"); /* Call*/\nwbTime_start(\"Copy\" /* String*/, \"Copying output memory to the CPU\"); /* Call*/\ncudaMemcpy(hostOutput /* Identifier*/, deviceOutput /* Identifier*/, byteSize /* Identifier*/, cudaMemcpyDeviceToHost); /* Call*/\nwbTime_stop(\"Copy\" /* String*/, \"Copying output memory to the CPU\"); /* Call*/\nwbTime_start(\"GPU\" /* String*/, \"Freeing GPU Memory\"); /* Call*/\nwbTime_stop(\"GPU\" /* String*/, \"Freeing GPU Memory\"); /* Call*/\nwbSolution(args /* Identifier*/, hostOutput /* Identifier*/, inputLength); /* Call*/\nfree(hostInput1); /* Call*/\nfree(hostInput2); /* Call*/\nfree(hostOutput); /* Call*/\nreturn 0; /* Return*/\n}\n",
                "loc": {
                    "end": {
                        "column": 1,
                        "line": 85
                    },
                    "start": {
                        "column": 1,
                        "line": 11
                    }
                },
                "raw": ";\r\n    float * hostInput2;\r\n    float * hostOutput;\r\n    float * deviceInput1;\r\n    float * deviceInput2;\r\n    float * deviceOutput;\r\n\r\n    args = wbArg_read(argc, argv);\r\n\r\n    wbTime_start(Generic, \"Importing data and creating memory on host\");\r\n    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);\r\n    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);\r\n    hostOutput = (float *) malloc(inputLength * sizeof(float));\r\n    wbTime_stop(Generic, \"Importing data and creating memory on host\");\r\n\r\n    wbLog(TRACE, \"The input length is \", inputLength, \" elements\");\r\n\r\n\r\n    wbTime_start(GPU, \"Allocating GPU memory.\");\r\n    //@@ Allocate GPU memory here\r\n    int byteSize =sizeof(float) * inputLength;\r\n\r\n    wbTime_stop(GPU, \"Allocating GPU memory.\");\r\n\r\n    wbTime_start(GPU, \"Copying input memory to the GPU.\");\r\n    //@@ Copy memory to the GPU here\r\n\r\n    cudaMalloc((void **) &deviceInput1, byteSize);\r\n    cudaMalloc((void **) &deviceInput2, byteSize);\r\n    cudaMalloc((void **) &deviceOutput, byteSize);\r\n\r\n\r\n    wbTime_stop(GPU, \"Copying input memory to the GPU.\");\r\n\r\n    //@@ Initialize the grid and block dimensions here\r\n    cudaMemcpy(deviceInput1, hostInput1, byteSize,cudaMemcpyHostToDevice);\r\n\r\n    cudaMemcpy(deviceInput2, hostInput1, byteSize,cudaMemcpyHostToDevice);\r\n\r\n\r\n    wbTime_start(Compute, \"Performing CUDA computation\");\r\n    //@@ Launch the GPU Kernel here\r\n     int block_size = 16;\r\n     int n_blocks = inputLength /block_size + (inputLength%block_size == 0 ? 0:1);\r\n\r\n\r\n    vecAdd<<< n_blocks, block_size>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);\r\n\r\n\r\n    cudaThreadSynchronize();\r\n    wbTime_stop(Compute, \"Performing CUDA computation\");\r\n\r\n    wbTime_start(Copy, \"Copying output memory to the CPU\");\r\n    //@@ Copy the GPU memory back to the CPU here\r\n    cudaMemcpy(hostOutput, deviceOutput, byteSize,cudaMemcpyDeviceToHost);\r\n\r\n    wbTime_stop(Copy, \"Copying output memory to the CPU\");\r\n\r\n    wbTime_start(GPU, \"Freeing GPU Memory\");\r\n    //@@ Free the GPU memory here\r\n\r\n\r\n    wbTime_stop(GPU, \"Freeing GPU Memory\");\r\n\r\n    wbSolution(args, hostOutput, inputLength);\r\n\r\n    free(hostInput1);\r\n    free(hostInput2);\r\n    free(hostOutput);\r\n\r\n    return 0;\r\n}...",
                "type": "BlockStatement"
            },
            "cform": "int  main(int  argc /* Parameter*/, char ** argv){\nint  args; /* Declare*/\nint  inputLength; /* Declare*/\nfloat * hostInput1; /* Declare*/\nfloat * hostInput2; /* Declare*/\nfloat * hostOutput; /* Declare*/\nfloat * deviceInput1; /* Declare*/\nfloat * deviceInput2; /* Declare*/\nfloat * deviceOutput; /* Declare*/\nargs = wbArg_read(argc /* Identifier*/, argv); /* Assign*/\nwbTime_start(\"Generic\" /* String*/, \"Importing data and creating memory on host\"); /* Call*/\nhostInput1 = wbImport(\"input0\" /* String*/, & inputLength); /* Assign*/\nhostInput2 = wbImport(\"input1\" /* String*/, & inputLength); /* Assign*/\nhostOutput = malloc(inputLength * sizeof(float )); /* Assign*/\nwbTime_stop(\"Generic\" /* String*/, \"Importing data and creating memory on host\"); /* Call*/\nwbLog(\"TRACE\" /* String*/, \"The input length is \" /* String*/, inputLength /* Identifier*/, \" elements\"); /* Call*/\nwbTime_start(\"GPU\" /* String*/, \"Allocating GPU memory.\"); /* Call*/\nint  byteSize = sizeof(float ) * inputLength; /* Declare*/\nwbTime_stop(\"GPU\" /* String*/, \"Allocating GPU memory.\"); /* Call*/\nwbTime_start(\"GPU\" /* String*/, \"Copying input memory to the GPU.\"); /* Call*/\ncudaMalloc(& deviceInput1 /* UnaryOperator*/, byteSize); /* Call*/\ncudaMalloc(& deviceInput2 /* UnaryOperator*/, byteSize); /* Call*/\ncudaMalloc(& deviceOutput /* UnaryOperator*/, byteSize); /* Call*/\nwbTime_stop(\"GPU\" /* String*/, \"Copying input memory to the GPU.\"); /* Call*/\ncudaMemcpy(deviceInput1 /* Identifier*/, hostInput1 /* Identifier*/, byteSize /* Identifier*/, cudaMemcpyHostToDevice); /* Call*/\ncudaMemcpy(deviceInput2 /* Identifier*/, hostInput1 /* Identifier*/, byteSize /* Identifier*/, cudaMemcpyHostToDevice); /* Call*/\nwbTime_start(\"Compute\" /* String*/, \"Performing CUDA computation\"); /* Call*/\nint  block_size = 16; /* Declare*/\nint  n_blocks = inputLength / block_size + (inputLength % block_size == 0 ? 0 : 1); /* Declare*/\nvecAdd<<<{n_blocks} /* CompoundNode*/, {block_size}>>>(deviceInput1 /* Identifier*/, deviceInput2 /* Identifier*/, deviceOutput /* Identifier*/, inputLength); /* Call*/\ncudaThreadSynchronize(); /* Call*/\nwbTime_stop(\"Compute\" /* String*/, \"Performing CUDA computation\"); /* Call*/\nwbTime_start(\"Copy\" /* String*/, \"Copying output memory to the CPU\"); /* Call*/\ncudaMemcpy(hostOutput /* Identifier*/, deviceOutput /* Identifier*/, byteSize /* Identifier*/, cudaMemcpyDeviceToHost); /* Call*/\nwbTime_stop(\"Copy\" /* String*/, \"Copying output memory to the CPU\"); /* Call*/\nwbTime_start(\"GPU\" /* String*/, \"Freeing GPU Memory\"); /* Call*/\nwbTime_stop(\"GPU\" /* String*/, \"Freeing GPU Memory\"); /* Call*/\nwbSolution(args /* Identifier*/, hostOutput /* Identifier*/, inputLength); /* Call*/\nfree(hostInput1); /* Call*/\nfree(hostInput2); /* Call*/\nfree(hostOutput); /* Call*/\nreturn 0; /* Return*/\n}\n",
            "id": "main",
            "loc": {
                "end": {
                    "column": 1,
                    "line": 85
                },
                "start": {
                    "column": 1,
                    "line": 11
                }
            },
            "params": [
                {
                    "data": {
                        "cform": "argc",
                        "loc": {
                            "end": {
                                "column": 14,
                                "line": 11
                            },
                            "start": {
                                "column": 10,
                                "line": 11
                            }
                        },
                        "name": "argc",
                        "raw": "int argc",
                        "type": "Identifier"
                    },
                    "kind": {
                        "address_spaces": [],
                        "bases": [
                            {
                                "cform": "int",
                                "loc": {
                                    "end": {
                                        "column": 14,
                                        "line": 11
                                    },
                                    "start": {
                                        "column": 10,
                                        "line": 11
                                    }
                                },
                                "raw": "int",
                                "type": "Literal",
                                "value": "int"
                            }
                        ],
                        "cform": "int ",
                        "loc": {
                            "end": {
                                "column": 14,
                                "line": 11
                            },
                            "start": {
                                "column": 10,
                                "line": 11
                            }
                        },
                        "qualifiers": [],
                        "raw": "int argc",
                        "type": "TypeSpecification"
                    },
                    "type": "ParameterExpression"
                },
                {
                    "data": {
                        "cform": "argv",
                        "loc": {
                            "end": {
                                "column": 28,
                                "line": 11
                            },
                            "start": {
                                "column": 20,
                                "line": 11
                            }
                        },
                        "name": "argv",
                        "raw": "char ** argv",
                        "type": "Identifier"
                    },
                    "kind": {
                        "cform": "char **",
                        "loc": {
                            "end": {
                                "column": 28,
                                "line": 11
                            },
                            "start": {
                                "column": 20,
                                "line": 11
                            }
                        },
                        "raw": "char ** argv",
                        "type": "ReferenceType",
                        "value": {
                            "cform": "char *",
                            "loc": {
                                "end": {
                                    "column": 28,
                                    "line": 11
                                },
                                "start": {
                                    "column": 20,
                                    "line": 11
                                }
                            },
                            "raw": "char ** argv",
                            "type": "ReferenceType",
                            "value": {
                                "address_spaces": [],
                                "bases": [
                                    {
                                        "cform": "char",
                                        "loc": {
                                            "end": {
                                                "column": 28,
                                                "line": 11
                                            },
                                            "start": {
                                                "column": 20,
                                                "line": 11
                                            }
                                        },
                                        "raw": "char",
                                        "type": "Literal",
                                        "value": "char"
                                    }
                                ],
                                "cform": "char ",
                                "loc": {
                                    "end": {
                                        "column": 28,
                                        "line": 11
                                    },
                                    "start": {
                                        "column": 20,
                                        "line": 11
                                    }
                                },
                                "qualifiers": [],
                                "raw": "char ** argv",
                                "type": "TypeSpecification"
                            }
                        }
                    },
                    "type": "ParameterExpression"
                }
            ],
            "raw": ";\r\n    float * hostInput2;\r\n    float * hostOutput;\r\n    float * deviceInput1;\r\n    float * deviceInput2;\r\n    float * deviceOutput;\r\n\r\n    args = wbArg_read(argc, argv);\r\n\r\n    wbTime_start(Generic, \"Importing data and creating memory on host\");\r\n    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);\r\n    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);\r\n    hostOutput = (float *) malloc(inputLength * sizeof(float));\r\n    wbTime_stop(Generic, \"Importing data and creating memory on host\");\r\n\r\n    wbLog(TRACE, \"The input length is \", inputLength, \" elements\");\r\n\r\n\r\n    wbTime_start(GPU, \"Allocating GPU memory.\");\r\n    //@@ Allocate GPU memory here\r\n    int byteSize =sizeof(float) * inputLength;\r\n\r\n    wbTime_stop(GPU, \"Allocating GPU memory.\");\r\n\r\n    wbTime_start(GPU, \"Copying input memory to the GPU.\");\r\n    //@@ Copy memory to the GPU here\r\n\r\n    cudaMalloc((void **) &deviceInput1, byteSize);\r\n    cudaMalloc((void **) &deviceInput2, byteSize);\r\n    cudaMalloc((void **) &deviceOutput, byteSize);\r\n\r\n\r\n    wbTime_stop(GPU, \"Copying input memory to the GPU.\");\r\n\r\n    //@@ Initialize the grid and block dimensions here\r\n    cudaMemcpy(deviceInput1, hostInput1, byteSize,cudaMemcpyHostToDevice);\r\n\r\n    cudaMemcpy(deviceInput2, hostInput1, byteSize,cudaMemcpyHostToDevice);\r\n\r\n\r\n    wbTime_start(Compute, \"Performing CUDA computation\");\r\n    //@@ Launch the GPU Kernel here\r\n     int block_size = 16;\r\n     int n_blocks = inputLength /block_size + (inputLength%block_size == 0 ? 0:1);\r\n\r\n\r\n    vecAdd<<< n_blocks, block_size>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);\r\n\r\n\r\n    cudaThreadSynchronize();\r\n    wbTime_stop(Compute, \"Performing CUDA computation\");\r\n\r\n    wbTime_start(Copy, \"Copying output memory to the CPU\");\r\n    //@@ Copy the GPU memory back to the CPU here\r\n    cudaMemcpy(hostOutput, deviceOutput, byteSize,cudaMemcpyDeviceToHost);\r\n\r\n    wbTime_stop(Copy, \"Copying output memory to the CPU\");\r\n\r\n    wbTime_start(GPU, \"Freeing GPU Memory\");\r\n    //@@ Free the GPU memory here\r\n\r\n\r\n    wbTime_stop(GPU, \"Freeing GPU Memory\");\r\n\r\n    wbSolution(args, hostOutput, inputLength);\r\n\r\n    free(hostInput1);\r\n    free(hostInput2);\r\n    free(hostOutput);\r\n\r\n    return 0;\r\n}...",
            "type": "Function"
        }
    ],
    "cform": "__global__ void  vecAdd(float * in1 /* Parameter*/, float * in2 /* Parameter*/, float * out /* Parameter*/, int  len){\nint  idx = blockIdx.x * blockDim.x + threadIdx.x; /* Declare*/\nif (idx < len){\nout[idx] = in1[idx] + in2[idx]; /* Assign*/\n}\n}\nint  main(int  argc /* Parameter*/, char ** argv){\nint  args; /* Declare*/\nint  inputLength; /* Declare*/\nfloat * hostInput1; /* Declare*/\nfloat * hostInput2; /* Declare*/\nfloat * hostOutput; /* Declare*/\nfloat * deviceInput1; /* Declare*/\nfloat * deviceInput2; /* Declare*/\nfloat * deviceOutput; /* Declare*/\nargs = wbArg_read(argc /* Identifier*/, argv); /* Assign*/\nwbTime_start(\"Generic\" /* String*/, \"Importing data and creating memory on host\"); /* Call*/\nhostInput1 = wbImport(\"input0\" /* String*/, & inputLength); /* Assign*/\nhostInput2 = wbImport(\"input1\" /* String*/, & inputLength); /* Assign*/\nhostOutput = malloc(inputLength * sizeof(float )); /* Assign*/\nwbTime_stop(\"Generic\" /* String*/, \"Importing data and creating memory on host\"); /* Call*/\nwbLog(\"TRACE\" /* String*/, \"The input length is \" /* String*/, inputLength /* Identifier*/, \" elements\"); /* Call*/\nwbTime_start(\"GPU\" /* String*/, \"Allocating GPU memory.\"); /* Call*/\nint  byteSize = sizeof(float ) * inputLength; /* Declare*/\nwbTime_stop(\"GPU\" /* String*/, \"Allocating GPU memory.\"); /* Call*/\nwbTime_start(\"GPU\" /* String*/, \"Copying input memory to the GPU.\"); /* Call*/\ncudaMalloc(& deviceInput1 /* UnaryOperator*/, byteSize); /* Call*/\ncudaMalloc(& deviceInput2 /* UnaryOperator*/, byteSize); /* Call*/\ncudaMalloc(& deviceOutput /* UnaryOperator*/, byteSize); /* Call*/\nwbTime_stop(\"GPU\" /* String*/, \"Copying input memory to the GPU.\"); /* Call*/\ncudaMemcpy(deviceInput1 /* Identifier*/, hostInput1 /* Identifier*/, byteSize /* Identifier*/, cudaMemcpyHostToDevice); /* Call*/\ncudaMemcpy(deviceInput2 /* Identifier*/, hostInput1 /* Identifier*/, byteSize /* Identifier*/, cudaMemcpyHostToDevice); /* Call*/\nwbTime_start(\"Compute\" /* String*/, \"Performing CUDA computation\"); /* Call*/\nint  block_size = 16; /* Declare*/\nint  n_blocks = inputLength / block_size + (inputLength % block_size == 0 ? 0 : 1); /* Declare*/\nvecAdd<<<{n_blocks} /* CompoundNode*/, {block_size}>>>(deviceInput1 /* Identifier*/, deviceInput2 /* Identifier*/, deviceOutput /* Identifier*/, inputLength); /* Call*/\ncudaThreadSynchronize(); /* Call*/\nwbTime_stop(\"Compute\" /* String*/, \"Performing CUDA computation\"); /* Call*/\nwbTime_start(\"Copy\" /* String*/, \"Copying output memory to the CPU\"); /* Call*/\ncudaMemcpy(hostOutput /* Identifier*/, deviceOutput /* Identifier*/, byteSize /* Identifier*/, cudaMemcpyDeviceToHost); /* Call*/\nwbTime_stop(\"Copy\" /* String*/, \"Copying output memory to the CPU\"); /* Call*/\nwbTime_start(\"GPU\" /* String*/, \"Freeing GPU Memory\"); /* Call*/\nwbTime_stop(\"GPU\" /* String*/, \"Freeing GPU Memory\"); /* Call*/\nwbSolution(args /* Identifier*/, hostOutput /* Identifier*/, inputLength); /* Call*/\nfree(hostInput1); /* Call*/\nfree(hostInput2); /* Call*/\nfree(hostOutput); /* Call*/\nreturn 0; /* Return*/\n}\n",
    "loc": {
        "end": {
            "column": 0,
            "line": 0
        },
        "start": {
            "column": 0,
            "line": 0
        }
    },
    "raw": "",
    "type": "Program"
}
;
        export var mp2:any = {
            "body": [{
                "attributes": ["__global__"],
                "body": {
                    "body": [{
                        "cform": "int  row = blockIdx.y * blockDim.y + threadIdx.y",
                        "declarations": [{
                            "cform": "int  row = blockIdx.y * blockDim.y + threadIdx.y",
                            "id": {
                                "cform": "row",
                                "loc": {"end": {"column": 49, "line": 9}, "start": {"column": 3, "line": 9}},
                                "name": "row",
                                "raw": "int row = blockIdx.y * blockDim.y + threadIdx.y",
                                "type": "Identifier"
                            },
                            "init": {
                                "cform": "blockIdx.y * blockDim.y + threadIdx.y",
                                "left": {
                                    "cform": "blockIdx.y * blockDim.y",
                                    "left": {
                                        "cform": "blockIdx.y",
                                        "left": {
                                            "cform": "blockIdx",
                                            "kind": {
                                                "address_spaces": [],
                                                "bases": [{
                                                    "cform": "uint3",
                                                    "loc": {
                                                        "end": {"column": 31, "line": 22},
                                                        "start": {"column": 1, "line": 22}
                                                    },
                                                    "raw": "uint3",
                                                    "type": "Literal",
                                                    "value": "uint3"
                                                }],
                                                "cform": "const uint3 ",
                                                "loc": {
                                                    "end": {"column": 31, "line": 22},
                                                    "start": {"column": 1, "line": 22}
                                                },
                                                "qualifiers": [{
                                                    "cform": "const",
                                                    "loc": {
                                                        "end": {"column": 31, "line": 22},
                                                        "start": {"column": 1, "line": 22}
                                                    },
                                                    "raw": "uint3 __device__ extern const blockIdx",
                                                    "type": "Literal",
                                                    "value": "const"
                                                }],
                                                "raw": "uint3 __device__ extern const blockIdx",
                                                "type": "TypeSpecification"
                                            },
                                            "loc": {
                                                "end": {"column": 31, "line": 22},
                                                "start": {"column": 1, "line": 22}
                                            },
                                            "name": "blockIdx",
                                            "raw": "uint3 __device__ extern const blockIdx",
                                            "type": "Identifier"
                                        },
                                        "loc": {"end": {"column": 22, "line": 9}, "start": {"column": 13, "line": 9}},
                                        "operator": ".",
                                        "raw": "blockIdx.y",
                                        "right": {
                                            "cform": "y",
                                            "kind": {
                                                "address_spaces": [],
                                                "bases": [{
                                                    "cform": "unsigned int",
                                                    "loc": {
                                                        "end": {"column": 21, "line": 12},
                                                        "start": {"column": 5, "line": 12}
                                                    },
                                                    "raw": "unsigned int",
                                                    "type": "Literal",
                                                    "value": "unsigned int"
                                                }],
                                                "cform": "unsigned int ",
                                                "loc": {
                                                    "end": {"column": 21, "line": 12},
                                                    "start": {"column": 5, "line": 12}
                                                },
                                                "qualifiers": [],
                                                "raw": "unsigned int x, y",
                                                "type": "TypeSpecification"
                                            },
                                            "loc": {
                                                "end": {"column": 21, "line": 12},
                                                "start": {"column": 5, "line": 12}
                                            },
                                            "name": "y",
                                            "raw": "unsigned int x, y",
                                            "type": "Identifier"
                                        },
                                        "type": "MemberExpression"
                                    },
                                    "loc": {"end": {"column": 35, "line": 9}, "start": {"column": 13, "line": 9}},
                                    "operator": "*",
                                    "raw": "blockIdx.y * blockDim.y",
                                    "right": {
                                        "cform": "blockDim.y",
                                        "left": {
                                            "cform": "blockDim",
                                            "kind": {
                                                "address_spaces": [],
                                                "bases": [{
                                                    "cform": "struct dim3",
                                                    "loc": {
                                                        "end": {"column": 30, "line": 23},
                                                        "start": {"column": 1, "line": 23}
                                                    },
                                                    "raw": "struct dim3",
                                                    "type": "Literal",
                                                    "value": "struct dim3"
                                                }],
                                                "cform": "const struct dim3 ",
                                                "loc": {
                                                    "end": {"column": 30, "line": 23},
                                                    "start": {"column": 1, "line": 23}
                                                },
                                                "qualifiers": [{
                                                    "cform": "const",
                                                    "loc": {
                                                        "end": {"column": 30, "line": 23},
                                                        "start": {"column": 1, "line": 23}
                                                    },
                                                    "raw": "dim3 __device__ extern const blockDim",
                                                    "type": "Literal",
                                                    "value": "const"
                                                }],
                                                "raw": "dim3 __device__ extern const blockDim",
                                                "type": "TypeSpecification"
                                            },
                                            "loc": {
                                                "end": {"column": 30, "line": 23},
                                                "start": {"column": 1, "line": 23}
                                            },
                                            "name": "blockDim",
                                            "raw": "dim3 __device__ extern const blockDim",
                                            "type": "Identifier"
                                        },
                                        "loc": {"end": {"column": 35, "line": 9}, "start": {"column": 26, "line": 9}},
                                        "operator": ".",
                                        "raw": "blockDim.y",
                                        "right": {
                                            "cform": "y",
                                            "kind": {
                                                "address_spaces": [],
                                                "bases": [{
                                                    "cform": "unsigned int",
                                                    "loc": {
                                                        "end": {"column": 19, "line": 16},
                                                        "start": {"column": 3, "line": 16}
                                                    },
                                                    "raw": "unsigned int",
                                                    "type": "Literal",
                                                    "value": "unsigned int"
                                                }],
                                                "cform": "unsigned int ",
                                                "loc": {
                                                    "end": {"column": 19, "line": 16},
                                                    "start": {"column": 3, "line": 16}
                                                },
                                                "qualifiers": [],
                                                "raw": "unsigned int x, y",
                                                "type": "TypeSpecification"
                                            },
                                            "loc": {
                                                "end": {"column": 19, "line": 16},
                                                "start": {"column": 3, "line": 16}
                                            },
                                            "name": "y",
                                            "raw": "unsigned int x, y",
                                            "type": "Identifier"
                                        },
                                        "type": "MemberExpression"
                                    },
                                    "type": "BinaryExpression"
                                },
                                "loc": {"end": {"column": 49, "line": 9}, "start": {"column": 13, "line": 9}},
                                "operator": "+",
                                "raw": "blockIdx.y * blockDim.y + threadIdx.y",
                                "right": {
                                    "cform": "threadIdx.y",
                                    "left": {
                                        "cform": "threadIdx",
                                        "kind": {
                                            "address_spaces": [],
                                            "bases": [{
                                                "cform": "uint3",
                                                "loc": {
                                                    "end": {"column": 31, "line": 21},
                                                    "start": {"column": 1, "line": 21}
                                                },
                                                "raw": "uint3",
                                                "type": "Literal",
                                                "value": "uint3"
                                            }],
                                            "cform": "const uint3 ",
                                            "loc": {
                                                "end": {"column": 31, "line": 21},
                                                "start": {"column": 1, "line": 21}
                                            },
                                            "qualifiers": [{
                                                "cform": "const",
                                                "loc": {
                                                    "end": {"column": 31, "line": 21},
                                                    "start": {"column": 1, "line": 21}
                                                },
                                                "raw": "uint3 __device__ extern const threadIdx",
                                                "type": "Literal",
                                                "value": "const"
                                            }],
                                            "raw": "uint3 __device__ extern const threadIdx",
                                            "type": "TypeSpecification"
                                        },
                                        "loc": {"end": {"column": 31, "line": 21}, "start": {"column": 1, "line": 21}},
                                        "name": "threadIdx",
                                        "raw": "uint3 __device__ extern const threadIdx",
                                        "type": "Identifier"
                                    },
                                    "loc": {"end": {"column": 49, "line": 9}, "start": {"column": 39, "line": 9}},
                                    "operator": ".",
                                    "raw": "threadIdx.y",
                                    "right": {
                                        "cform": "y",
                                        "kind": {
                                            "address_spaces": [],
                                            "bases": [{
                                                "cform": "unsigned int",
                                                "loc": {
                                                    "end": {"column": 21, "line": 12},
                                                    "start": {"column": 5, "line": 12}
                                                },
                                                "raw": "unsigned int",
                                                "type": "Literal",
                                                "value": "unsigned int"
                                            }],
                                            "cform": "unsigned int ",
                                            "loc": {
                                                "end": {"column": 21, "line": 12},
                                                "start": {"column": 5, "line": 12}
                                            },
                                            "qualifiers": [],
                                            "raw": "unsigned int x, y",
                                            "type": "TypeSpecification"
                                        },
                                        "loc": {"end": {"column": 21, "line": 12}, "start": {"column": 5, "line": 12}},
                                        "name": "y",
                                        "raw": "unsigned int x, y",
                                        "type": "Identifier"
                                    },
                                    "type": "MemberExpression"
                                },
                                "type": "BinaryExpression"
                            },
                            "loc": {"end": {"column": 49, "line": 9}, "start": {"column": 3, "line": 9}},
                            "raw": "int row = blockIdx.y * blockDim.y + threadIdx.y",
                            "type": "VariableDeclarator"
                        }],
                        "loc": {"end": {"column": 49, "line": 9}, "start": {"column": 3, "line": 9}},
                        "raw": "int row = blockIdx.y * blockDim.y + threadIdx.y",
                        "type": "VariableDeclaration"
                    }, {
                        "cform": "int  col = blockIdx.x * blockDim.x + threadIdx.x",
                        "declarations": [{
                            "cform": "int  col = blockIdx.x * blockDim.x + threadIdx.x",
                            "id": {
                                "cform": "col",
                                "loc": {"end": {"column": 49, "line": 10}, "start": {"column": 3, "line": 10}},
                                "name": "col",
                                "raw": "int col = blockIdx.x * blockDim.x + threadIdx.x",
                                "type": "Identifier"
                            },
                            "init": {
                                "cform": "blockIdx.x * blockDim.x + threadIdx.x",
                                "left": {
                                    "cform": "blockIdx.x * blockDim.x",
                                    "left": {
                                        "cform": "blockIdx.x",
                                        "left": {
                                            "cform": "blockIdx",
                                            "kind": {
                                                "address_spaces": [],
                                                "bases": [{
                                                    "cform": "uint3",
                                                    "loc": {
                                                        "end": {"column": 31, "line": 22},
                                                        "start": {"column": 1, "line": 22}
                                                    },
                                                    "raw": "uint3",
                                                    "type": "Literal",
                                                    "value": "uint3"
                                                }],
                                                "cform": "const uint3 ",
                                                "loc": {
                                                    "end": {"column": 31, "line": 22},
                                                    "start": {"column": 1, "line": 22}
                                                },
                                                "qualifiers": [{
                                                    "cform": "const",
                                                    "loc": {
                                                        "end": {"column": 31, "line": 22},
                                                        "start": {"column": 1, "line": 22}
                                                    },
                                                    "raw": "uint3 __device__ extern const blockIdx",
                                                    "type": "Literal",
                                                    "value": "const"
                                                }],
                                                "raw": "uint3 __device__ extern const blockIdx",
                                                "type": "TypeSpecification"
                                            },
                                            "loc": {
                                                "end": {"column": 31, "line": 22},
                                                "start": {"column": 1, "line": 22}
                                            },
                                            "name": "blockIdx",
                                            "raw": "uint3 __device__ extern const blockIdx",
                                            "type": "Identifier"
                                        },
                                        "loc": {"end": {"column": 22, "line": 10}, "start": {"column": 13, "line": 10}},
                                        "operator": ".",
                                        "raw": "blockIdx.x",
                                        "right": {
                                            "cform": "x",
                                            "kind": {
                                                "address_spaces": [],
                                                "bases": [{
                                                    "cform": "unsigned int",
                                                    "loc": {
                                                        "end": {"column": 18, "line": 12},
                                                        "start": {"column": 5, "line": 12}
                                                    },
                                                    "raw": "unsigned int",
                                                    "type": "Literal",
                                                    "value": "unsigned int"
                                                }],
                                                "cform": "unsigned int ",
                                                "loc": {
                                                    "end": {"column": 18, "line": 12},
                                                    "start": {"column": 5, "line": 12}
                                                },
                                                "qualifiers": [],
                                                "raw": "unsigned int x",
                                                "type": "TypeSpecification"
                                            },
                                            "loc": {
                                                "end": {"column": 18, "line": 12},
                                                "start": {"column": 5, "line": 12}
                                            },
                                            "name": "x",
                                            "raw": "unsigned int x",
                                            "type": "Identifier"
                                        },
                                        "type": "MemberExpression"
                                    },
                                    "loc": {"end": {"column": 35, "line": 10}, "start": {"column": 13, "line": 10}},
                                    "operator": "*",
                                    "raw": "blockIdx.x * blockDim.x",
                                    "right": {
                                        "cform": "blockDim.x",
                                        "left": {
                                            "cform": "blockDim",
                                            "kind": {
                                                "address_spaces": [],
                                                "bases": [{
                                                    "cform": "struct dim3",
                                                    "loc": {
                                                        "end": {"column": 30, "line": 23},
                                                        "start": {"column": 1, "line": 23}
                                                    },
                                                    "raw": "struct dim3",
                                                    "type": "Literal",
                                                    "value": "struct dim3"
                                                }],
                                                "cform": "const struct dim3 ",
                                                "loc": {
                                                    "end": {"column": 30, "line": 23},
                                                    "start": {"column": 1, "line": 23}
                                                },
                                                "qualifiers": [{
                                                    "cform": "const",
                                                    "loc": {
                                                        "end": {"column": 30, "line": 23},
                                                        "start": {"column": 1, "line": 23}
                                                    },
                                                    "raw": "dim3 __device__ extern const blockDim",
                                                    "type": "Literal",
                                                    "value": "const"
                                                }],
                                                "raw": "dim3 __device__ extern const blockDim",
                                                "type": "TypeSpecification"
                                            },
                                            "loc": {
                                                "end": {"column": 30, "line": 23},
                                                "start": {"column": 1, "line": 23}
                                            },
                                            "name": "blockDim",
                                            "raw": "dim3 __device__ extern const blockDim",
                                            "type": "Identifier"
                                        },
                                        "loc": {"end": {"column": 35, "line": 10}, "start": {"column": 26, "line": 10}},
                                        "operator": ".",
                                        "raw": "blockDim.x",
                                        "right": {
                                            "cform": "x",
                                            "kind": {
                                                "address_spaces": [],
                                                "bases": [{
                                                    "cform": "unsigned int",
                                                    "loc": {
                                                        "end": {"column": 16, "line": 16},
                                                        "start": {"column": 3, "line": 16}
                                                    },
                                                    "raw": "unsigned int",
                                                    "type": "Literal",
                                                    "value": "unsigned int"
                                                }],
                                                "cform": "unsigned int ",
                                                "loc": {
                                                    "end": {"column": 16, "line": 16},
                                                    "start": {"column": 3, "line": 16}
                                                },
                                                "qualifiers": [],
                                                "raw": "unsigned int x",
                                                "type": "TypeSpecification"
                                            },
                                            "loc": {
                                                "end": {"column": 16, "line": 16},
                                                "start": {"column": 3, "line": 16}
                                            },
                                            "name": "x",
                                            "raw": "unsigned int x",
                                            "type": "Identifier"
                                        },
                                        "type": "MemberExpression"
                                    },
                                    "type": "BinaryExpression"
                                },
                                "loc": {"end": {"column": 49, "line": 10}, "start": {"column": 13, "line": 10}},
                                "operator": "+",
                                "raw": "blockIdx.x * blockDim.x + threadIdx.x",
                                "right": {
                                    "cform": "threadIdx.x",
                                    "left": {
                                        "cform": "threadIdx",
                                        "kind": {
                                            "address_spaces": [],
                                            "bases": [{
                                                "cform": "uint3",
                                                "loc": {
                                                    "end": {"column": 31, "line": 21},
                                                    "start": {"column": 1, "line": 21}
                                                },
                                                "raw": "uint3",
                                                "type": "Literal",
                                                "value": "uint3"
                                            }],
                                            "cform": "const uint3 ",
                                            "loc": {
                                                "end": {"column": 31, "line": 21},
                                                "start": {"column": 1, "line": 21}
                                            },
                                            "qualifiers": [{
                                                "cform": "const",
                                                "loc": {
                                                    "end": {"column": 31, "line": 21},
                                                    "start": {"column": 1, "line": 21}
                                                },
                                                "raw": "uint3 __device__ extern const threadIdx",
                                                "type": "Literal",
                                                "value": "const"
                                            }],
                                            "raw": "uint3 __device__ extern const threadIdx",
                                            "type": "TypeSpecification"
                                        },
                                        "loc": {"end": {"column": 31, "line": 21}, "start": {"column": 1, "line": 21}},
                                        "name": "threadIdx",
                                        "raw": "uint3 __device__ extern const threadIdx",
                                        "type": "Identifier"
                                    },
                                    "loc": {"end": {"column": 49, "line": 10}, "start": {"column": 39, "line": 10}},
                                    "operator": ".",
                                    "raw": "threadIdx.x",
                                    "right": {
                                        "cform": "x",
                                        "kind": {
                                            "address_spaces": [],
                                            "bases": [{
                                                "cform": "unsigned int",
                                                "loc": {
                                                    "end": {"column": 18, "line": 12},
                                                    "start": {"column": 5, "line": 12}
                                                },
                                                "raw": "unsigned int",
                                                "type": "Literal",
                                                "value": "unsigned int"
                                            }],
                                            "cform": "unsigned int ",
                                            "loc": {
                                                "end": {"column": 18, "line": 12},
                                                "start": {"column": 5, "line": 12}
                                            },
                                            "qualifiers": [],
                                            "raw": "unsigned int x",
                                            "type": "TypeSpecification"
                                        },
                                        "loc": {"end": {"column": 18, "line": 12}, "start": {"column": 5, "line": 12}},
                                        "name": "x",
                                        "raw": "unsigned int x",
                                        "type": "Identifier"
                                    },
                                    "type": "MemberExpression"
                                },
                                "type": "BinaryExpression"
                            },
                            "loc": {"end": {"column": 49, "line": 10}, "start": {"column": 3, "line": 10}},
                            "raw": "int col = blockIdx.x * blockDim.x + threadIdx.x",
                            "type": "VariableDeclarator"
                        }],
                        "loc": {"end": {"column": 49, "line": 10}, "start": {"column": 3, "line": 10}},
                        "raw": "int col = blockIdx.x * blockDim.x + threadIdx.x",
                        "type": "VariableDeclaration"
                    }, {
                        "cform": "if (row < numARows && col < numBColumns){\nfloat  sum = 0; /* Declare*/\nfor (int  ii = 0,ii < numAColumns,++ ii) {\nsum = A[row * numAColumns + ii] * B[ii * numBColumns + col]; /* Assign*/\n}\nC[row * numBColumns + col] = sum; /* Assign*/\n}\n",
                        "consequent": {
                            "body": [{
                                "cform": "float  sum = 0",
                                "declarations": [{
                                    "cform": "float  sum = 0",
                                    "id": {
                                        "cform": "sum",
                                        "loc": {"end": {"column": 17, "line": 12}, "start": {"column": 5, "line": 12}},
                                        "name": "sum",
                                        "raw": "float sum = 0",
                                        "type": "Identifier"
                                    },
                                    "init": {
                                        "cform": "0",
                                        "loc": {"end": {"column": 17, "line": 12}, "start": {"column": 17, "line": 12}},
                                        "raw": "0",
                                        "type": "Integer32Literal",
                                        "value": 0
                                    },
                                    "loc": {"end": {"column": 17, "line": 12}, "start": {"column": 5, "line": 12}},
                                    "raw": "float sum = 0",
                                    "type": "VariableDeclarator"
                                }],
                                "loc": {"end": {"column": 17, "line": 12}, "start": {"column": 5, "line": 12}},
                                "raw": "float sum = 0",
                                "type": "VariableDeclaration"
                            }, {
                                "body": {
                                    "body": [{
                                        "cform": "sum = A[row * numAColumns + ii] * B[ii * numBColumns + col]",
                                        "left": {
                                            "cform": "sum",
                                            "kind": {
                                                "address_spaces": [],
                                                "bases": [{
                                                    "cform": "float",
                                                    "loc": {
                                                        "end": {"column": 17, "line": 12},
                                                        "start": {"column": 5, "line": 12}
                                                    },
                                                    "raw": "float",
                                                    "type": "Literal",
                                                    "value": "float"
                                                }],
                                                "cform": "float ",
                                                "loc": {
                                                    "end": {"column": 17, "line": 12},
                                                    "start": {"column": 5, "line": 12}
                                                },
                                                "qualifiers": [],
                                                "raw": "float sum = 0",
                                                "type": "TypeSpecification"
                                            },
                                            "loc": {
                                                "end": {"column": 17, "line": 12},
                                                "start": {"column": 5, "line": 12}
                                            },
                                            "name": "sum",
                                            "raw": "float sum = 0",
                                            "type": "Identifier"
                                        },
                                        "loc": {"end": {"column": 66, "line": 14}, "start": {"column": 7, "line": 14}},
                                        "operator": "=",
                                        "raw": "sum += A[row * numAColumns + ii] * B[ii * numBColumns + col]",
                                        "right": {
                                            "cform": "A[row * numAColumns + ii] * B[ii * numBColumns + col]",
                                            "left": {
                                                "cform": "A[row * numAColumns + ii]",
                                                "computed": true,
                                                "loc": {
                                                    "end": {"column": 38, "line": 14},
                                                    "start": {"column": 14, "line": 14}
                                                },
                                                "object": {
                                                    "cform": "A",
                                                    "kind": {
                                                        "cform": "float *",
                                                        "loc": {
                                                            "end": {"column": 30, "line": 6},
                                                            "start": {"column": 23, "line": 6}
                                                        },
                                                        "raw": "float *A",
                                                        "type": "ReferenceType",
                                                        "value": {
                                                            "address_spaces": [],
                                                            "bases": [{
                                                                "cform": "float",
                                                                "loc": {
                                                                    "end": {"column": 30, "line": 6},
                                                                    "start": {"column": 23, "line": 6}
                                                                },
                                                                "raw": "float",
                                                                "type": "Literal",
                                                                "value": "float"
                                                            }],
                                                            "cform": "float ",
                                                            "loc": {
                                                                "end": {"column": 30, "line": 6},
                                                                "start": {"column": 23, "line": 6}
                                                            },
                                                            "qualifiers": [],
                                                            "raw": "float *A",
                                                            "type": "TypeSpecification"
                                                        }
                                                    },
                                                    "loc": {
                                                        "end": {"column": 30, "line": 6},
                                                        "start": {"column": 23, "line": 6}
                                                    },
                                                    "name": "A",
                                                    "raw": "float *A",
                                                    "type": "Identifier"
                                                },
                                                "property": {
                                                    "cform": "row * numAColumns + ii",
                                                    "left": {
                                                        "cform": "row * numAColumns",
                                                        "left": {
                                                            "cform": "row",
                                                            "kind": {
                                                                "address_spaces": [],
                                                                "bases": [{
                                                                    "cform": "int",
                                                                    "loc": {
                                                                        "end": {"column": 49, "line": 9},
                                                                        "start": {"column": 3, "line": 9}
                                                                    },
                                                                    "raw": "int",
                                                                    "type": "Literal",
                                                                    "value": "int"
                                                                }],
                                                                "cform": "int ",
                                                                "loc": {
                                                                    "end": {"column": 49, "line": 9},
                                                                    "start": {"column": 3, "line": 9}
                                                                },
                                                                "qualifiers": [],
                                                                "raw": "int row = blockIdx.y * blockDim.y + threadIdx.y",
                                                                "type": "TypeSpecification"
                                                            },
                                                            "loc": {
                                                                "end": {"column": 49, "line": 9},
                                                                "start": {"column": 3, "line": 9}
                                                            },
                                                            "name": "row",
                                                            "raw": "int row = blockIdx.y * blockDim.y + threadIdx.y",
                                                            "type": "Identifier"
                                                        },
                                                        "loc": {
                                                            "end": {"column": 22, "line": 14},
                                                            "start": {"column": 16, "line": 14}
                                                        },
                                                        "operator": "*",
                                                        "raw": "row * numAColumns",
                                                        "right": {
                                                            "cform": "numAColumns",
                                                            "kind": {
                                                                "address_spaces": [],
                                                                "bases": [{
                                                                    "cform": "int",
                                                                    "loc": {
                                                                        "end": {"column": 27, "line": 7},
                                                                        "start": {"column": 23, "line": 7}
                                                                    },
                                                                    "raw": "int",
                                                                    "type": "Literal",
                                                                    "value": "int"
                                                                }],
                                                                "cform": "int ",
                                                                "loc": {
                                                                    "end": {"column": 27, "line": 7},
                                                                    "start": {"column": 23, "line": 7}
                                                                },
                                                                "qualifiers": [],
                                                                "raw": "int numAColumns",
                                                                "type": "TypeSpecification"
                                                            },
                                                            "loc": {
                                                                "end": {"column": 27, "line": 7},
                                                                "start": {"column": 23, "line": 7}
                                                            },
                                                            "name": "numAColumns",
                                                            "raw": "int numAColumns",
                                                            "type": "Identifier"
                                                        },
                                                        "type": "BinaryExpression"
                                                    },
                                                    "loc": {
                                                        "end": {"column": 36, "line": 14},
                                                        "start": {"column": 16, "line": 14}
                                                    },
                                                    "operator": "+",
                                                    "raw": "row * numAColumns + ii",
                                                    "right": {
                                                        "cform": "ii",
                                                        "kind": {
                                                            "address_spaces": [],
                                                            "bases": [{
                                                                "cform": "int",
                                                                "loc": {
                                                                    "end": {"column": 19, "line": 13},
                                                                    "start": {"column": 10, "line": 13}
                                                                },
                                                                "raw": "int",
                                                                "type": "Literal",
                                                                "value": "int"
                                                            }],
                                                            "cform": "int ",
                                                            "loc": {
                                                                "end": {"column": 19, "line": 13},
                                                                "start": {"column": 10, "line": 13}
                                                            },
                                                            "qualifiers": [],
                                                            "raw": "int ii = 0",
                                                            "type": "TypeSpecification"
                                                        },
                                                        "loc": {
                                                            "end": {"column": 19, "line": 13},
                                                            "start": {"column": 10, "line": 13}
                                                        },
                                                        "name": "ii",
                                                        "raw": "int ii = 0",
                                                        "type": "Identifier"
                                                    },
                                                    "type": "BinaryExpression"
                                                },
                                                "raw": "A[row * numAColumns + ii]",
                                                "type": "SubscriptExpression"
                                            },
                                            "loc": {
                                                "end": {"column": 66, "line": 14},
                                                "start": {"column": 14, "line": 14}
                                            },
                                            "operator": "*",
                                            "raw": "A[row * numAColumns + ii] * B[ii * numBColumns + col]",
                                            "right": {
                                                "cform": "B[ii * numBColumns + col]",
                                                "computed": true,
                                                "loc": {
                                                    "end": {"column": 66, "line": 14},
                                                    "start": {"column": 42, "line": 14}
                                                },
                                                "object": {
                                                    "cform": "B",
                                                    "kind": {
                                                        "cform": "float *",
                                                        "loc": {
                                                            "end": {"column": 40, "line": 6},
                                                            "start": {"column": 33, "line": 6}
                                                        },
                                                        "raw": "float *B",
                                                        "type": "ReferenceType",
                                                        "value": {
                                                            "address_spaces": [],
                                                            "bases": [{
                                                                "cform": "float",
                                                                "loc": {
                                                                    "end": {"column": 40, "line": 6},
                                                                    "start": {"column": 33, "line": 6}
                                                                },
                                                                "raw": "float",
                                                                "type": "Literal",
                                                                "value": "float"
                                                            }],
                                                            "cform": "float ",
                                                            "loc": {
                                                                "end": {"column": 40, "line": 6},
                                                                "start": {"column": 33, "line": 6}
                                                            },
                                                            "qualifiers": [],
                                                            "raw": "float *B",
                                                            "type": "TypeSpecification"
                                                        }
                                                    },
                                                    "loc": {
                                                        "end": {"column": 40, "line": 6},
                                                        "start": {"column": 33, "line": 6}
                                                    },
                                                    "name": "B",
                                                    "raw": "float *B",
                                                    "type": "Identifier"
                                                },
                                                "property": {
                                                    "cform": "ii * numBColumns + col",
                                                    "left": {
                                                        "cform": "ii * numBColumns",
                                                        "left": {
                                                            "cform": "ii",
                                                            "kind": {
                                                                "address_spaces": [],
                                                                "bases": [{
                                                                    "cform": "int",
                                                                    "loc": {
                                                                        "end": {"column": 19, "line": 13},
                                                                        "start": {"column": 10, "line": 13}
                                                                    },
                                                                    "raw": "int",
                                                                    "type": "Literal",
                                                                    "value": "int"
                                                                }],
                                                                "cform": "int ",
                                                                "loc": {
                                                                    "end": {"column": 19, "line": 13},
                                                                    "start": {"column": 10, "line": 13}
                                                                },
                                                                "qualifiers": [],
                                                                "raw": "int ii = 0",
                                                                "type": "TypeSpecification"
                                                            },
                                                            "loc": {
                                                                "end": {"column": 19, "line": 13},
                                                                "start": {"column": 10, "line": 13}
                                                            },
                                                            "name": "ii",
                                                            "raw": "int ii = 0",
                                                            "type": "Identifier"
                                                        },
                                                        "loc": {
                                                            "end": {"column": 49, "line": 14},
                                                            "start": {"column": 44, "line": 14}
                                                        },
                                                        "operator": "*",
                                                        "raw": "ii * numBColumns",
                                                        "right": {
                                                            "cform": "numBColumns",
                                                            "kind": {
                                                                "address_spaces": [],
                                                                "bases": [{
                                                                    "cform": "int",
                                                                    "loc": {
                                                                        "end": {"column": 58, "line": 7},
                                                                        "start": {"column": 54, "line": 7}
                                                                    },
                                                                    "raw": "int",
                                                                    "type": "Literal",
                                                                    "value": "int"
                                                                }],
                                                                "cform": "int ",
                                                                "loc": {
                                                                    "end": {"column": 58, "line": 7},
                                                                    "start": {"column": 54, "line": 7}
                                                                },
                                                                "qualifiers": [],
                                                                "raw": "int numBColumns",
                                                                "type": "TypeSpecification"
                                                            },
                                                            "loc": {
                                                                "end": {"column": 58, "line": 7},
                                                                "start": {"column": 54, "line": 7}
                                                            },
                                                            "name": "numBColumns",
                                                            "raw": "int numBColumns",
                                                            "type": "Identifier"
                                                        },
                                                        "type": "BinaryExpression"
                                                    },
                                                    "loc": {
                                                        "end": {"column": 63, "line": 14},
                                                        "start": {"column": 44, "line": 14}
                                                    },
                                                    "operator": "+",
                                                    "raw": "ii * numBColumns + col",
                                                    "right": {
                                                        "cform": "col",
                                                        "kind": {
                                                            "address_spaces": [],
                                                            "bases": [{
                                                                "cform": "int",
                                                                "loc": {
                                                                    "end": {"column": 49, "line": 10},
                                                                    "start": {"column": 3, "line": 10}
                                                                },
                                                                "raw": "int",
                                                                "type": "Literal",
                                                                "value": "int"
                                                            }],
                                                            "cform": "int ",
                                                            "loc": {
                                                                "end": {"column": 49, "line": 10},
                                                                "start": {"column": 3, "line": 10}
                                                            },
                                                            "qualifiers": [],
                                                            "raw": "int col = blockIdx.x * blockDim.x + threadIdx.x",
                                                            "type": "TypeSpecification"
                                                        },
                                                        "loc": {
                                                            "end": {"column": 49, "line": 10},
                                                            "start": {"column": 3, "line": 10}
                                                        },
                                                        "name": "col",
                                                        "raw": "int col = blockIdx.x * blockDim.x + threadIdx.x",
                                                        "type": "Identifier"
                                                    },
                                                    "type": "BinaryExpression"
                                                },
                                                "raw": "B[ii * numBColumns + col]",
                                                "type": "SubscriptExpression"
                                            },
                                            "type": "BinaryExpression"
                                        },
                                        "type": "AssignmentExpression"
                                    }],
                                    "cform": "{\nsum = A[row * numAColumns + ii] * B[ii * numBColumns + col]; /* Assign*/\n}\n",
                                    "loc": {"end": {"column": 5, "line": 15}, "start": {"column": 5, "line": 13}},
                                    "raw": "mns + col];\n    }...",
                                    "type": "BlockStatement"
                                },
                                "cform": "for (int  ii = 0,ii < numAColumns,++ ii) {\nsum = A[row * numAColumns + ii] * B[ii * numBColumns + col]; /* Assign*/\n}\n",
                                "init": [{
                                    "cform": "int  ii = 0",
                                    "declarations": [{
                                        "cform": "int  ii = 0",
                                        "id": {
                                            "cform": "ii",
                                            "loc": {
                                                "end": {"column": 19, "line": 13},
                                                "start": {"column": 10, "line": 13}
                                            },
                                            "name": "ii",
                                            "raw": "int ii = 0",
                                            "type": "Identifier"
                                        },
                                        "init": {
                                            "cform": "0",
                                            "loc": {
                                                "end": {"column": 19, "line": 13},
                                                "start": {"column": 19, "line": 13}
                                            },
                                            "raw": "0",
                                            "type": "Integer32Literal",
                                            "value": 0
                                        },
                                        "loc": {"end": {"column": 19, "line": 13}, "start": {"column": 10, "line": 13}},
                                        "raw": "int ii = 0",
                                        "type": "VariableDeclarator"
                                    }],
                                    "loc": {"end": {"column": 19, "line": 13}, "start": {"column": 10, "line": 13}},
                                    "raw": "int ii = 0",
                                    "type": "VariableDeclaration"
                                }],
                                "loc": {"end": {"column": 5, "line": 15}, "start": {"column": 5, "line": 13}},
                                "raw": "mns + col];\n    }...",
                                "test": {
                                    "cform": "ii < numAColumns",
                                    "left": {
                                        "cform": "ii",
                                        "kind": {
                                            "address_spaces": [],
                                            "bases": [{
                                                "cform": "int",
                                                "loc": {
                                                    "end": {"column": 19, "line": 13},
                                                    "start": {"column": 10, "line": 13}
                                                },
                                                "raw": "int",
                                                "type": "Literal",
                                                "value": "int"
                                            }],
                                            "cform": "int ",
                                            "loc": {
                                                "end": {"column": 19, "line": 13},
                                                "start": {"column": 10, "line": 13}
                                            },
                                            "qualifiers": [],
                                            "raw": "int ii = 0",
                                            "type": "TypeSpecification"
                                        },
                                        "loc": {"end": {"column": 19, "line": 13}, "start": {"column": 10, "line": 13}},
                                        "name": "ii",
                                        "raw": "int ii = 0",
                                        "type": "Identifier"
                                    },
                                    "loc": {"end": {"column": 27, "line": 13}, "start": {"column": 22, "line": 13}},
                                    "operator": "<",
                                    "raw": "ii < numAColumns",
                                    "right": {
                                        "cform": "numAColumns",
                                        "kind": {
                                            "address_spaces": [],
                                            "bases": [{
                                                "cform": "int",
                                                "loc": {
                                                    "end": {"column": 27, "line": 7},
                                                    "start": {"column": 23, "line": 7}
                                                },
                                                "raw": "int",
                                                "type": "Literal",
                                                "value": "int"
                                            }],
                                            "cform": "int ",
                                            "loc": {
                                                "end": {"column": 27, "line": 7},
                                                "start": {"column": 23, "line": 7}
                                            },
                                            "qualifiers": [],
                                            "raw": "int numAColumns",
                                            "type": "TypeSpecification"
                                        },
                                        "loc": {"end": {"column": 27, "line": 7}, "start": {"column": 23, "line": 7}},
                                        "name": "numAColumns",
                                        "raw": "int numAColumns",
                                        "type": "Identifier"
                                    },
                                    "type": "BinaryExpression"
                                },
                                "type": "ForStatement",
                                "update": {
                                    "argument": {
                                        "cform": "ii",
                                        "kind": {
                                            "address_spaces": [],
                                            "bases": [{
                                                "cform": "int",
                                                "loc": {
                                                    "end": {"column": 19, "line": 13},
                                                    "start": {"column": 10, "line": 13}
                                                },
                                                "raw": "int",
                                                "type": "Literal",
                                                "value": "int"
                                            }],
                                            "cform": "int ",
                                            "loc": {
                                                "end": {"column": 19, "line": 13},
                                                "start": {"column": 10, "line": 13}
                                            },
                                            "qualifiers": [],
                                            "raw": "int ii = 0",
                                            "type": "TypeSpecification"
                                        },
                                        "loc": {"end": {"column": 19, "line": 13}, "start": {"column": 10, "line": 13}},
                                        "name": "ii",
                                        "raw": "int ii = 0",
                                        "type": "Identifier"
                                    },
                                    "cform": "++ ii",
                                    "loc": {"end": {"column": 42, "line": 13}, "start": {"column": 40, "line": 13}},
                                    "operator": "++",
                                    "prefix": true,
                                    "raw": "ii++",
                                    "type": "UnaryExpression"
                                }
                            }, {
                                "cform": "C[row * numBColumns + col] = sum",
                                "left": {
                                    "cform": "C[row * numBColumns + col]",
                                    "computed": true,
                                    "loc": {"end": {"column": 30, "line": 16}, "start": {"column": 5, "line": 16}},
                                    "object": {
                                        "cform": "C",
                                        "kind": {
                                            "cform": "float *",
                                            "loc": {
                                                "end": {"column": 50, "line": 6},
                                                "start": {"column": 43, "line": 6}
                                            },
                                            "raw": "float *C",
                                            "type": "ReferenceType",
                                            "value": {
                                                "address_spaces": [],
                                                "bases": [{
                                                    "cform": "float",
                                                    "loc": {
                                                        "end": {"column": 50, "line": 6},
                                                        "start": {"column": 43, "line": 6}
                                                    },
                                                    "raw": "float",
                                                    "type": "Literal",
                                                    "value": "float"
                                                }],
                                                "cform": "float ",
                                                "loc": {
                                                    "end": {"column": 50, "line": 6},
                                                    "start": {"column": 43, "line": 6}
                                                },
                                                "qualifiers": [],
                                                "raw": "float *C",
                                                "type": "TypeSpecification"
                                            }
                                        },
                                        "loc": {"end": {"column": 50, "line": 6}, "start": {"column": 43, "line": 6}},
                                        "name": "C",
                                        "raw": "float *C",
                                        "type": "Identifier"
                                    },
                                    "property": {
                                        "cform": "row * numBColumns + col",
                                        "left": {
                                            "cform": "row * numBColumns",
                                            "left": {
                                                "cform": "row",
                                                "kind": {
                                                    "address_spaces": [],
                                                    "bases": [{
                                                        "cform": "int",
                                                        "loc": {
                                                            "end": {"column": 49, "line": 9},
                                                            "start": {"column": 3, "line": 9}
                                                        },
                                                        "raw": "int",
                                                        "type": "Literal",
                                                        "value": "int"
                                                    }],
                                                    "cform": "int ",
                                                    "loc": {
                                                        "end": {"column": 49, "line": 9},
                                                        "start": {"column": 3, "line": 9}
                                                    },
                                                    "qualifiers": [],
                                                    "raw": "int row = blockIdx.y * blockDim.y + threadIdx.y",
                                                    "type": "TypeSpecification"
                                                },
                                                "loc": {
                                                    "end": {"column": 49, "line": 9},
                                                    "start": {"column": 3, "line": 9}
                                                },
                                                "name": "row",
                                                "raw": "int row = blockIdx.y * blockDim.y + threadIdx.y",
                                                "type": "Identifier"
                                            },
                                            "loc": {
                                                "end": {"column": 13, "line": 16},
                                                "start": {"column": 7, "line": 16}
                                            },
                                            "operator": "*",
                                            "raw": "row * numBColumns",
                                            "right": {
                                                "cform": "numBColumns",
                                                "kind": {
                                                    "address_spaces": [],
                                                    "bases": [{
                                                        "cform": "int",
                                                        "loc": {
                                                            "end": {"column": 58, "line": 7},
                                                            "start": {"column": 54, "line": 7}
                                                        },
                                                        "raw": "int",
                                                        "type": "Literal",
                                                        "value": "int"
                                                    }],
                                                    "cform": "int ",
                                                    "loc": {
                                                        "end": {"column": 58, "line": 7},
                                                        "start": {"column": 54, "line": 7}
                                                    },
                                                    "qualifiers": [],
                                                    "raw": "int numBColumns",
                                                    "type": "TypeSpecification"
                                                },
                                                "loc": {
                                                    "end": {"column": 58, "line": 7},
                                                    "start": {"column": 54, "line": 7}
                                                },
                                                "name": "numBColumns",
                                                "raw": "int numBColumns",
                                                "type": "Identifier"
                                            },
                                            "type": "BinaryExpression"
                                        },
                                        "loc": {"end": {"column": 27, "line": 16}, "start": {"column": 7, "line": 16}},
                                        "operator": "+",
                                        "raw": "row * numBColumns + col",
                                        "right": {
                                            "cform": "col",
                                            "kind": {
                                                "address_spaces": [],
                                                "bases": [{
                                                    "cform": "int",
                                                    "loc": {
                                                        "end": {"column": 49, "line": 10},
                                                        "start": {"column": 3, "line": 10}
                                                    },
                                                    "raw": "int",
                                                    "type": "Literal",
                                                    "value": "int"
                                                }],
                                                "cform": "int ",
                                                "loc": {
                                                    "end": {"column": 49, "line": 10},
                                                    "start": {"column": 3, "line": 10}
                                                },
                                                "qualifiers": [],
                                                "raw": "int col = blockIdx.x * blockDim.x + threadIdx.x",
                                                "type": "TypeSpecification"
                                            },
                                            "loc": {
                                                "end": {"column": 49, "line": 10},
                                                "start": {"column": 3, "line": 10}
                                            },
                                            "name": "col",
                                            "raw": "int col = blockIdx.x * blockDim.x + threadIdx.x",
                                            "type": "Identifier"
                                        },
                                        "type": "BinaryExpression"
                                    },
                                    "raw": "C[row * numBColumns + col]",
                                    "type": "SubscriptExpression"
                                },
                                "loc": {"end": {"column": 34, "line": 16}, "start": {"column": 5, "line": 16}},
                                "operator": "=",
                                "raw": "C[row * numBColumns + col] = sum",
                                "right": {
                                    "cform": "sum",
                                    "kind": {
                                        "address_spaces": [],
                                        "bases": [{
                                            "cform": "float",
                                            "loc": {
                                                "end": {"column": 17, "line": 12},
                                                "start": {"column": 5, "line": 12}
                                            },
                                            "raw": "float",
                                            "type": "Literal",
                                            "value": "float"
                                        }],
                                        "cform": "float ",
                                        "loc": {"end": {"column": 17, "line": 12}, "start": {"column": 5, "line": 12}},
                                        "qualifiers": [],
                                        "raw": "float sum = 0",
                                        "type": "TypeSpecification"
                                    },
                                    "loc": {"end": {"column": 17, "line": 12}, "start": {"column": 5, "line": 12}},
                                    "name": "sum",
                                    "raw": "float sum = 0",
                                    "type": "Identifier"
                                },
                                "type": "AssignmentExpression"
                            }],
                            "cform": "{\nfloat  sum = 0; /* Declare*/\nfor (int  ii = 0,ii < numAColumns,++ ii) {\nsum = A[row * numAColumns + ii] * B[ii * numBColumns + col]; /* Assign*/\n}\nC[row * numBColumns + col] = sum; /* Assign*/\n}\n",
                            "loc": {"end": {"column": 3, "line": 17}, "start": {"column": 3, "line": 11}},
                            "raw": "; ii++) {\n      sum += A[row * numAColumns + ii] * B[ii * numBColumns + col];\n    }\n    C[row * numBColumns + col] = sum;\n  }...",
                            "type": "BlockStatement"
                        },
                        "loc": {"end": {"column": 3, "line": 17}, "start": {"column": 3, "line": 11}},
                        "raw": "; ii++) {\n      sum += A[row * numAColumns + ii] * B[ii * numBColumns + col];\n    }\n    C[row * numBColumns + col] = sum;\n  }...",
                        "test": {
                            "cform": "row < numARows && col < numBColumns",
                            "left": {
                                "cform": "row < numARows",
                                "left": {
                                    "cform": "row",
                                    "kind": {
                                        "address_spaces": [],
                                        "bases": [{
                                            "cform": "int",
                                            "loc": {
                                                "end": {"column": 49, "line": 9},
                                                "start": {"column": 3, "line": 9}
                                            },
                                            "raw": "int",
                                            "type": "Literal",
                                            "value": "int"
                                        }],
                                        "cform": "int ",
                                        "loc": {"end": {"column": 49, "line": 9}, "start": {"column": 3, "line": 9}},
                                        "qualifiers": [],
                                        "raw": "int row = blockIdx.y * blockDim.y + threadIdx.y",
                                        "type": "TypeSpecification"
                                    },
                                    "loc": {"end": {"column": 49, "line": 9}, "start": {"column": 3, "line": 9}},
                                    "name": "row",
                                    "raw": "int row = blockIdx.y * blockDim.y + threadIdx.y",
                                    "type": "Identifier"
                                },
                                "loc": {"end": {"column": 13, "line": 11}, "start": {"column": 7, "line": 11}},
                                "operator": "<",
                                "raw": "row < numARows",
                                "right": {
                                    "cform": "numARows",
                                    "kind": {
                                        "address_spaces": [],
                                        "bases": [{
                                            "cform": "int",
                                            "loc": {
                                                "end": {"column": 57, "line": 6},
                                                "start": {"column": 53, "line": 6}
                                            },
                                            "raw": "int",
                                            "type": "Literal",
                                            "value": "int"
                                        }],
                                        "cform": "int ",
                                        "loc": {"end": {"column": 57, "line": 6}, "start": {"column": 53, "line": 6}},
                                        "qualifiers": [],
                                        "raw": "int numARows",
                                        "type": "TypeSpecification"
                                    },
                                    "loc": {"end": {"column": 57, "line": 6}, "start": {"column": 53, "line": 6}},
                                    "name": "numARows",
                                    "raw": "int numARows",
                                    "type": "Identifier"
                                },
                                "type": "BinaryExpression"
                            },
                            "loc": {"end": {"column": 31, "line": 11}, "start": {"column": 7, "line": 11}},
                            "operator": "&&",
                            "raw": "row < numARows && col < numBColumns",
                            "right": {
                                "cform": "col < numBColumns",
                                "left": {
                                    "cform": "col",
                                    "kind": {
                                        "address_spaces": [],
                                        "bases": [{
                                            "cform": "int",
                                            "loc": {
                                                "end": {"column": 49, "line": 10},
                                                "start": {"column": 3, "line": 10}
                                            },
                                            "raw": "int",
                                            "type": "Literal",
                                            "value": "int"
                                        }],
                                        "cform": "int ",
                                        "loc": {"end": {"column": 49, "line": 10}, "start": {"column": 3, "line": 10}},
                                        "qualifiers": [],
                                        "raw": "int col = blockIdx.x * blockDim.x + threadIdx.x",
                                        "type": "TypeSpecification"
                                    },
                                    "loc": {"end": {"column": 49, "line": 10}, "start": {"column": 3, "line": 10}},
                                    "name": "col",
                                    "raw": "int col = blockIdx.x * blockDim.x + threadIdx.x",
                                    "type": "Identifier"
                                },
                                "loc": {"end": {"column": 31, "line": 11}, "start": {"column": 25, "line": 11}},
                                "operator": "<",
                                "raw": "col < numBColumns",
                                "right": {
                                    "cform": "numBColumns",
                                    "kind": {
                                        "address_spaces": [],
                                        "bases": [{
                                            "cform": "int",
                                            "loc": {
                                                "end": {"column": 58, "line": 7},
                                                "start": {"column": 54, "line": 7}
                                            },
                                            "raw": "int",
                                            "type": "Literal",
                                            "value": "int"
                                        }],
                                        "cform": "int ",
                                        "loc": {"end": {"column": 58, "line": 7}, "start": {"column": 54, "line": 7}},
                                        "qualifiers": [],
                                        "raw": "int numBColumns",
                                        "type": "TypeSpecification"
                                    },
                                    "loc": {"end": {"column": 58, "line": 7}, "start": {"column": 54, "line": 7}},
                                    "name": "numBColumns",
                                    "raw": "int numBColumns",
                                    "type": "Identifier"
                                },
                                "type": "BinaryExpression"
                            },
                            "type": "BinaryExpression"
                        },
                        "type": "IfStatement"
                    }],
                    "cform": "{\nint  row = blockIdx.y * blockDim.y + threadIdx.y; /* Declare*/\nint  col = blockIdx.x * blockDim.x + threadIdx.x; /* Declare*/\nif (row < numARows && col < numBColumns){\nfloat  sum = 0; /* Declare*/\nfor (int  ii = 0,ii < numAColumns,++ ii) {\nsum = A[row * numAColumns + ii] * B[ii * numBColumns + col]; /* Assign*/\n}\nC[row * numBColumns + col] = sum; /* Assign*/\n}\n}\n",
                    "loc": {"end": {"column": 1, "line": 18}, "start": {"column": 1, "line": 6}},
                    "raw": "",
                    "type": "BlockStatement"
                },
                "cform": "__global__ void  sgemm(float * A /* Parameter*/, float * B /* Parameter*/, float * C /* Parameter*/, int  numARows /* Parameter*/, int  numAColumns /* Parameter*/, int  numBRows /* Parameter*/, int  numBColumns){\nint  row = blockIdx.y * blockDim.y + threadIdx.y; /* Declare*/\nint  col = blockIdx.x * blockDim.x + threadIdx.x; /* Declare*/\nif (row < numARows && col < numBColumns){\nfloat  sum = 0; /* Declare*/\nfor (int  ii = 0,ii < numAColumns,++ ii) {\nsum = A[row * numAColumns + ii] * B[ii * numBColumns + col]; /* Assign*/\n}\nC[row * numBColumns + col] = sum; /* Assign*/\n}\n}\n",
                "id": "sgemm",
                "loc": {"end": {"column": 1, "line": 18}, "start": {"column": 1, "line": 6}},
                "params": [{
                    "data": {
                        "cform": "A",
                        "loc": {"end": {"column": 30, "line": 6}, "start": {"column": 23, "line": 6}},
                        "name": "A",
                        "raw": "float *A",
                        "type": "Identifier"
                    }, "type": "ParameterExpression"
                }, {
                    "data": {
                        "cform": "B",
                        "loc": {"end": {"column": 40, "line": 6}, "start": {"column": 33, "line": 6}},
                        "name": "B",
                        "raw": "float *B",
                        "type": "Identifier"
                    }, "type": "ParameterExpression"
                }, {
                    "data": {
                        "cform": "C",
                        "loc": {"end": {"column": 50, "line": 6}, "start": {"column": 43, "line": 6}},
                        "name": "C",
                        "raw": "float *C",
                        "type": "Identifier"
                    }, "type": "ParameterExpression"
                }, {
                    "data": {
                        "cform": "numARows",
                        "loc": {"end": {"column": 57, "line": 6}, "start": {"column": 53, "line": 6}},
                        "name": "numARows",
                        "raw": "int numARows",
                        "type": "Identifier"
                    }, "type": "ParameterExpression"
                }, {
                    "data": {
                        "cform": "numAColumns",
                        "loc": {"end": {"column": 27, "line": 7}, "start": {"column": 23, "line": 7}},
                        "name": "numAColumns",
                        "raw": "int numAColumns",
                        "type": "Identifier"
                    }, "type": "ParameterExpression"
                }, {
                    "data": {
                        "cform": "numBRows",
                        "loc": {"end": {"column": 44, "line": 7}, "start": {"column": 40, "line": 7}},
                        "name": "numBRows",
                        "raw": "int numBRows",
                        "type": "Identifier"
                    }, "type": "ParameterExpression"
                }, {
                    "data": {
                        "cform": "numBColumns",
                        "loc": {"end": {"column": 58, "line": 7}, "start": {"column": 54, "line": 7}},
                        "name": "numBColumns",
                        "raw": "int numBColumns",
                        "type": "Identifier"
                    }, "type": "ParameterExpression"
                }],
                "raw": "",
                "type": "Function"
            }, {
                "attributes": [],
                "body": {
                    "body": [{
                        "cform": "int  args",
                        "declarations": [{
                            "cform": "int  args",
                            "id": {
                                "cform": "args",
                                "loc": {"end": {"column": 11, "line": 23}, "start": {"column": 3, "line": 23}},
                                "name": "args",
                                "raw": "wbArg_t args",
                                "type": "Identifier"
                            },
                            "loc": {"end": {"column": 11, "line": 23}, "start": {"column": 3, "line": 23}},
                            "raw": "wbArg_t args",
                            "type": "VariableDeclarator"
                        }],
                        "loc": {"end": {"column": 11, "line": 23}, "start": {"column": 3, "line": 23}},
                        "raw": "wbArg_t args",
                        "type": "VariableDeclaration"
                    }, {
                        "cform": "float * hostA",
                        "declarations": [{
                            "cform": "float * hostA",
                            "id": {
                                "cform": "hostA",
                                "loc": {"end": {"column": 10, "line": 24}, "start": {"column": 3, "line": 24}},
                                "name": "hostA",
                                "raw": "float *hostA",
                                "type": "Identifier"
                            },
                            "loc": {"end": {"column": 10, "line": 24}, "start": {"column": 3, "line": 24}},
                            "raw": "float *hostA",
                            "type": "VariableDeclarator"
                        }],
                        "loc": {"end": {"column": 10, "line": 24}, "start": {"column": 3, "line": 24}},
                        "raw": "float *hostA",
                        "type": "VariableDeclaration"
                    }, {
                        "cform": "float * hostB",
                        "declarations": [{
                            "cform": "float * hostB",
                            "id": {
                                "cform": "hostB",
                                "loc": {"end": {"column": 10, "line": 25}, "start": {"column": 3, "line": 25}},
                                "name": "hostB",
                                "raw": "float *hostB",
                                "type": "Identifier"
                            },
                            "loc": {"end": {"column": 10, "line": 25}, "start": {"column": 3, "line": 25}},
                            "raw": "float *hostB",
                            "type": "VariableDeclarator"
                        }],
                        "loc": {"end": {"column": 10, "line": 25}, "start": {"column": 3, "line": 25}},
                        "raw": "float *hostB",
                        "type": "VariableDeclaration"
                    }, {
                        "cform": "float * hostC",
                        "declarations": [{
                            "cform": "float * hostC",
                            "id": {
                                "cform": "hostC",
                                "loc": {"end": {"column": 10, "line": 26}, "start": {"column": 3, "line": 26}},
                                "name": "hostC",
                                "raw": "float *hostC",
                                "type": "Identifier"
                            },
                            "loc": {"end": {"column": 10, "line": 26}, "start": {"column": 3, "line": 26}},
                            "raw": "float *hostC",
                            "type": "VariableDeclarator"
                        }],
                        "loc": {"end": {"column": 10, "line": 26}, "start": {"column": 3, "line": 26}},
                        "raw": "float *hostC",
                        "type": "VariableDeclaration"
                    }, {
                        "cform": "float * deviceA",
                        "declarations": [{
                            "cform": "float * deviceA",
                            "id": {
                                "cform": "deviceA",
                                "loc": {"end": {"column": 10, "line": 27}, "start": {"column": 3, "line": 27}},
                                "name": "deviceA",
                                "raw": "float *deviceA",
                                "type": "Identifier"
                            },
                            "loc": {"end": {"column": 10, "line": 27}, "start": {"column": 3, "line": 27}},
                            "raw": "float *deviceA",
                            "type": "VariableDeclarator"
                        }],
                        "loc": {"end": {"column": 10, "line": 27}, "start": {"column": 3, "line": 27}},
                        "raw": "float *deviceA",
                        "type": "VariableDeclaration"
                    }, {
                        "cform": "float * deviceB",
                        "declarations": [{
                            "cform": "float * deviceB",
                            "id": {
                                "cform": "deviceB",
                                "loc": {"end": {"column": 10, "line": 28}, "start": {"column": 3, "line": 28}},
                                "name": "deviceB",
                                "raw": "float *deviceB",
                                "type": "Identifier"
                            },
                            "loc": {"end": {"column": 10, "line": 28}, "start": {"column": 3, "line": 28}},
                            "raw": "float *deviceB",
                            "type": "VariableDeclarator"
                        }],
                        "loc": {"end": {"column": 10, "line": 28}, "start": {"column": 3, "line": 28}},
                        "raw": "float *deviceB",
                        "type": "VariableDeclaration"
                    }, {
                        "cform": "float * deviceC",
                        "declarations": [{
                            "cform": "float * deviceC",
                            "id": {
                                "cform": "deviceC",
                                "loc": {"end": {"column": 10, "line": 29}, "start": {"column": 3, "line": 29}},
                                "name": "deviceC",
                                "raw": "float *deviceC",
                                "type": "Identifier"
                            },
                            "loc": {"end": {"column": 10, "line": 29}, "start": {"column": 3, "line": 29}},
                            "raw": "float *deviceC",
                            "type": "VariableDeclarator"
                        }],
                        "loc": {"end": {"column": 10, "line": 29}, "start": {"column": 3, "line": 29}},
                        "raw": "float *deviceC",
                        "type": "VariableDeclaration"
                    }, {
                        "cform": "int  numARows",
                        "declarations": [{
                            "cform": "int  numARows",
                            "id": {
                                "cform": "numARows",
                                "loc": {"end": {"column": 7, "line": 30}, "start": {"column": 3, "line": 30}},
                                "name": "numARows",
                                "raw": "int numARows",
                                "type": "Identifier"
                            },
                            "loc": {"end": {"column": 7, "line": 30}, "start": {"column": 3, "line": 30}},
                            "raw": "int numARows",
                            "type": "VariableDeclarator"
                        }],
                        "loc": {"end": {"column": 7, "line": 30}, "start": {"column": 3, "line": 30}},
                        "raw": "int numARows",
                        "type": "VariableDeclaration"
                    }, {
                        "cform": "int  numAColumns",
                        "declarations": [{
                            "cform": "int  numAColumns",
                            "id": {
                                "cform": "numAColumns",
                                "loc": {"end": {"column": 7, "line": 31}, "start": {"column": 3, "line": 31}},
                                "name": "numAColumns",
                                "raw": "int numAColumns",
                                "type": "Identifier"
                            },
                            "loc": {"end": {"column": 7, "line": 31}, "start": {"column": 3, "line": 31}},
                            "raw": "int numAColumns",
                            "type": "VariableDeclarator"
                        }],
                        "loc": {"end": {"column": 7, "line": 31}, "start": {"column": 3, "line": 31}},
                        "raw": "int numAColumns",
                        "type": "VariableDeclaration"
                    }, {
                        "cform": "int  numBRows",
                        "declarations": [{
                            "cform": "int  numBRows",
                            "id": {
                                "cform": "numBRows",
                                "loc": {"end": {"column": 7, "line": 32}, "start": {"column": 3, "line": 32}},
                                "name": "numBRows",
                                "raw": "int numBRows",
                                "type": "Identifier"
                            },
                            "loc": {"end": {"column": 7, "line": 32}, "start": {"column": 3, "line": 32}},
                            "raw": "int numBRows",
                            "type": "VariableDeclarator"
                        }],
                        "loc": {"end": {"column": 7, "line": 32}, "start": {"column": 3, "line": 32}},
                        "raw": "int numBRows",
                        "type": "VariableDeclaration"
                    }, {
                        "cform": "int  numBColumns",
                        "declarations": [{
                            "cform": "int  numBColumns",
                            "id": {
                                "cform": "numBColumns",
                                "loc": {"end": {"column": 7, "line": 33}, "start": {"column": 3, "line": 33}},
                                "name": "numBColumns",
                                "raw": "int numBColumns",
                                "type": "Identifier"
                            },
                            "loc": {"end": {"column": 7, "line": 33}, "start": {"column": 3, "line": 33}},
                            "raw": "int numBColumns",
                            "type": "VariableDeclarator"
                        }],
                        "loc": {"end": {"column": 7, "line": 33}, "start": {"column": 3, "line": 33}},
                        "raw": "int numBColumns",
                        "type": "VariableDeclaration"
                    }, {
                        "cform": "int  numCRows",
                        "declarations": [{
                            "cform": "int  numCRows",
                            "id": {
                                "cform": "numCRows",
                                "loc": {"end": {"column": 7, "line": 34}, "start": {"column": 3, "line": 34}},
                                "name": "numCRows",
                                "raw": "int numCRows",
                                "type": "Identifier"
                            },
                            "loc": {"end": {"column": 7, "line": 34}, "start": {"column": 3, "line": 34}},
                            "raw": "int numCRows",
                            "type": "VariableDeclarator"
                        }],
                        "loc": {"end": {"column": 7, "line": 34}, "start": {"column": 3, "line": 34}},
                        "raw": "int numCRows",
                        "type": "VariableDeclaration"
                    }, {
                        "cform": "int  numCColumns",
                        "declarations": [{
                            "cform": "int  numCColumns",
                            "id": {
                                "cform": "numCColumns",
                                "loc": {"end": {"column": 7, "line": 35}, "start": {"column": 3, "line": 35}},
                                "name": "numCColumns",
                                "raw": "int numCColumns",
                                "type": "Identifier"
                            },
                            "loc": {"end": {"column": 7, "line": 35}, "start": {"column": 3, "line": 35}},
                            "raw": "int numCColumns",
                            "type": "VariableDeclarator"
                        }],
                        "loc": {"end": {"column": 7, "line": 35}, "start": {"column": 3, "line": 35}},
                        "raw": "int numCColumns",
                        "type": "VariableDeclaration"
                    }, {
                        "cform": "args = wbArg_read(argc /* Identifier*/, argv)",
                        "left": {
                            "cform": "args",
                            "kind": {
                                "address_spaces": [],
                                "bases": [{
                                    "cform": "int",
                                    "loc": {"end": {"column": 11, "line": 23}, "start": {"column": 3, "line": 23}},
                                    "raw": "int",
                                    "type": "Literal",
                                    "value": "int"
                                }],
                                "cform": "int ",
                                "loc": {"end": {"column": 11, "line": 23}, "start": {"column": 3, "line": 23}},
                                "qualifiers": [],
                                "raw": "wbArg_t args",
                                "type": "TypeSpecification"
                            },
                            "loc": {"end": {"column": 11, "line": 23}, "start": {"column": 3, "line": 23}},
                            "name": "args",
                            "raw": "wbArg_t args",
                            "type": "Identifier"
                        },
                        "loc": {"end": {"column": 31, "line": 37}, "start": {"column": 3, "line": 37}},
                        "operator": "=",
                        "raw": "args = wbArg_read(argc, argv)",
                        "right": {
                            "arguments": [{
                                "cform": "argc",
                                "kind": {
                                    "address_spaces": [],
                                    "bases": [{
                                        "cform": "int",
                                        "loc": {"end": {"column": 14, "line": 22}, "start": {"column": 10, "line": 22}},
                                        "raw": "int",
                                        "type": "Literal",
                                        "value": "int"
                                    }],
                                    "cform": "int ",
                                    "loc": {"end": {"column": 14, "line": 22}, "start": {"column": 10, "line": 22}},
                                    "qualifiers": [],
                                    "raw": "int argc",
                                    "type": "TypeSpecification"
                                },
                                "loc": {"end": {"column": 14, "line": 22}, "start": {"column": 10, "line": 22}},
                                "name": "argc",
                                "raw": "int argc",
                                "type": "Identifier"
                            }, {
                                "cform": "argv",
                                "kind": {
                                    "cform": "char **",
                                    "loc": {"end": {"column": 27, "line": 22}, "start": {"column": 20, "line": 22}},
                                    "raw": "char **argv",
                                    "type": "ReferenceType",
                                    "value": {
                                        "cform": "char *",
                                        "loc": {"end": {"column": 27, "line": 22}, "start": {"column": 20, "line": 22}},
                                        "raw": "char **argv",
                                        "type": "ReferenceType",
                                        "value": {
                                            "address_spaces": [],
                                            "bases": [{
                                                "cform": "char",
                                                "loc": {
                                                    "end": {"column": 27, "line": 22},
                                                    "start": {"column": 20, "line": 22}
                                                },
                                                "raw": "char",
                                                "type": "Literal",
                                                "value": "char"
                                            }],
                                            "cform": "char ",
                                            "loc": {
                                                "end": {"column": 27, "line": 22},
                                                "start": {"column": 20, "line": 22}
                                            },
                                            "qualifiers": [],
                                            "raw": "char **argv",
                                            "type": "TypeSpecification"
                                        }
                                    }
                                },
                                "loc": {"end": {"column": 27, "line": 22}, "start": {"column": 20, "line": 22}},
                                "name": "argv",
                                "raw": "char **argv",
                                "type": "Identifier"
                            }],
                            "callee": {
                                "cform": "wbArg_read",
                                "loc": {"end": {"column": 31, "line": 37}, "start": {"column": 10, "line": 37}},
                                "name": "wbArg_read",
                                "raw": "wbArg_read(argc, argv)",
                                "type": "Identifier"
                            },
                            "cform": "wbArg_read(argc /* Identifier*/, argv)",
                            "loc": {"end": {"column": 31, "line": 37}, "start": {"column": 10, "line": 37}},
                            "raw": "wbArg_read(argc, argv)",
                            "type": "CallExpression"
                        },
                        "type": "AssignmentExpression"
                    }, {
                        "arguments": [{
                            "cform": "\"Generic\"",
                            "loc": {"end": {"column": 3, "line": 39}, "start": {"column": 3, "line": 39}},
                            "raw": "Generic",
                            "type": "StringLiteral",
                            "value": "\"Generic\""
                        }, {
                            "cform": "\"Importing data and creating memory on host\"",
                            "loc": {"end": {"column": 3, "line": 39}, "start": {"column": 3, "line": 39}},
                            "raw": "Importing data and creating memory on host",
                            "type": "StringLiteral",
                            "value": "\"Importing data and creating memory on host\""
                        }],
                        "callee": {
                            "cform": "wbTime_start",
                            "loc": {"end": {"column": 3, "line": 39}, "start": {"column": 3, "line": 39}},
                            "name": "wbTime_start",
                            "raw": "wbTime_start(#kind, __VA_ARGS__)",
                            "type": "Identifier"
                        },
                        "cform": "wbTime_start(\"Generic\" /* String*/, \"Importing data and creating memory on host\")",
                        "loc": {"end": {"column": 3, "line": 39}, "start": {"column": 3, "line": 39}},
                        "raw": "wbTime_start(#kind, __VA_ARGS__)",
                        "type": "CallExpression"
                    }, {}, {
                        "cform": "hostA = wbImport(\"input0\" /* String*/, & numARows /* UnaryOperator*/, & numAColumns)",
                        "left": {
                            "cform": "hostA",
                            "kind": {
                                "cform": "float *",
                                "loc": {"end": {"column": 10, "line": 24}, "start": {"column": 3, "line": 24}},
                                "raw": "float *hostA",
                                "type": "ReferenceType",
                                "value": {
                                    "address_spaces": [],
                                    "bases": [{
                                        "cform": "float",
                                        "loc": {"end": {"column": 10, "line": 24}, "start": {"column": 3, "line": 24}},
                                        "raw": "float",
                                        "type": "Literal",
                                        "value": "float"
                                    }],
                                    "cform": "float ",
                                    "loc": {"end": {"column": 10, "line": 24}, "start": {"column": 3, "line": 24}},
                                    "qualifiers": [],
                                    "raw": "float *hostA",
                                    "type": "TypeSpecification"
                                }
                            },
                            "loc": {"end": {"column": 10, "line": 24}, "start": {"column": 3, "line": 24}},
                            "name": "hostA",
                            "raw": "float *hostA",
                            "type": "Identifier"
                        },
                        "loc": {"end": {"column": 79, "line": 41}, "start": {"column": 3, "line": 40}},
                        "operator": "=",
                        "raw": "hostA =\n      ( float * )wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns)",
                        "right": {
                            "arguments": [{
                                "cform": "\"input0\"",
                                "loc": {"end": {"column": 27, "line": 41}, "start": {"column": 27, "line": 41}},
                                "raw": "input0",
                                "type": "StringLiteral",
                                "value": "\"input0\""
                            }, {
                                "argument": {
                                    "cform": "numARows",
                                    "kind": {
                                        "address_spaces": [],
                                        "bases": [{
                                            "cform": "int",
                                            "loc": {
                                                "end": {"column": 7, "line": 30},
                                                "start": {"column": 3, "line": 30}
                                            },
                                            "raw": "int",
                                            "type": "Literal",
                                            "value": "int"
                                        }],
                                        "cform": "int ",
                                        "loc": {"end": {"column": 7, "line": 30}, "start": {"column": 3, "line": 30}},
                                        "qualifiers": [],
                                        "raw": "int numARows",
                                        "type": "TypeSpecification"
                                    },
                                    "loc": {"end": {"column": 7, "line": 30}, "start": {"column": 3, "line": 30}},
                                    "name": "numARows",
                                    "raw": "int numARows",
                                    "type": "Identifier"
                                },
                                "cform": "& numARows",
                                "loc": {"end": {"column": 57, "line": 41}, "start": {"column": 56, "line": 41}},
                                "operator": "&",
                                "prefix": true,
                                "raw": "&numARows",
                                "type": "UnaryExpression"
                            }, {
                                "argument": {
                                    "cform": "numAColumns",
                                    "kind": {
                                        "address_spaces": [],
                                        "bases": [{
                                            "cform": "int",
                                            "loc": {
                                                "end": {"column": 7, "line": 31},
                                                "start": {"column": 3, "line": 31}
                                            },
                                            "raw": "int",
                                            "type": "Literal",
                                            "value": "int"
                                        }],
                                        "cform": "int ",
                                        "loc": {"end": {"column": 7, "line": 31}, "start": {"column": 3, "line": 31}},
                                        "qualifiers": [],
                                        "raw": "int numAColumns",
                                        "type": "TypeSpecification"
                                    },
                                    "loc": {"end": {"column": 7, "line": 31}, "start": {"column": 3, "line": 31}},
                                    "name": "numAColumns",
                                    "raw": "int numAColumns",
                                    "type": "Identifier"
                                },
                                "cform": "& numAColumns",
                                "loc": {"end": {"column": 68, "line": 41}, "start": {"column": 67, "line": 41}},
                                "operator": "&",
                                "prefix": true,
                                "raw": "&numAColumns",
                                "type": "UnaryExpression"
                            }],
                            "callee": {
                                "cform": "wbImport",
                                "loc": {"end": {"column": 79, "line": 41}, "start": {"column": 18, "line": 41}},
                                "name": "wbImport",
                                "raw": "wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns)",
                                "type": "Identifier"
                            },
                            "cform": "wbImport(\"input0\" /* String*/, & numARows /* UnaryOperator*/, & numAColumns)",
                            "loc": {"end": {"column": 79, "line": 41}, "start": {"column": 18, "line": 41}},
                            "raw": "wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns)",
                            "type": "CallExpression"
                        },
                        "type": "AssignmentExpression"
                    }, {
                        "cform": "hostB = wbImport(\"input1\" /* String*/, & numBRows /* UnaryOperator*/, & numBColumns)",
                        "left": {
                            "cform": "hostB",
                            "kind": {
                                "cform": "float *",
                                "loc": {"end": {"column": 10, "line": 25}, "start": {"column": 3, "line": 25}},
                                "raw": "float *hostB",
                                "type": "ReferenceType",
                                "value": {
                                    "address_spaces": [],
                                    "bases": [{
                                        "cform": "float",
                                        "loc": {"end": {"column": 10, "line": 25}, "start": {"column": 3, "line": 25}},
                                        "raw": "float",
                                        "type": "Literal",
                                        "value": "float"
                                    }],
                                    "cform": "float ",
                                    "loc": {"end": {"column": 10, "line": 25}, "start": {"column": 3, "line": 25}},
                                    "qualifiers": [],
                                    "raw": "float *hostB",
                                    "type": "TypeSpecification"
                                }
                            },
                            "loc": {"end": {"column": 10, "line": 25}, "start": {"column": 3, "line": 25}},
                            "name": "hostB",
                            "raw": "float *hostB",
                            "type": "Identifier"
                        },
                        "loc": {"end": {"column": 79, "line": 43}, "start": {"column": 3, "line": 42}},
                        "operator": "=",
                        "raw": "hostB =\n      ( float * )wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns)",
                        "right": {
                            "arguments": [{
                                "cform": "\"input1\"",
                                "loc": {"end": {"column": 27, "line": 43}, "start": {"column": 27, "line": 43}},
                                "raw": "input1",
                                "type": "StringLiteral",
                                "value": "\"input1\""
                            }, {
                                "argument": {
                                    "cform": "numBRows",
                                    "kind": {
                                        "address_spaces": [],
                                        "bases": [{
                                            "cform": "int",
                                            "loc": {
                                                "end": {"column": 7, "line": 32},
                                                "start": {"column": 3, "line": 32}
                                            },
                                            "raw": "int",
                                            "type": "Literal",
                                            "value": "int"
                                        }],
                                        "cform": "int ",
                                        "loc": {"end": {"column": 7, "line": 32}, "start": {"column": 3, "line": 32}},
                                        "qualifiers": [],
                                        "raw": "int numBRows",
                                        "type": "TypeSpecification"
                                    },
                                    "loc": {"end": {"column": 7, "line": 32}, "start": {"column": 3, "line": 32}},
                                    "name": "numBRows",
                                    "raw": "int numBRows",
                                    "type": "Identifier"
                                },
                                "cform": "& numBRows",
                                "loc": {"end": {"column": 57, "line": 43}, "start": {"column": 56, "line": 43}},
                                "operator": "&",
                                "prefix": true,
                                "raw": "&numBRows",
                                "type": "UnaryExpression"
                            }, {
                                "argument": {
                                    "cform": "numBColumns",
                                    "kind": {
                                        "address_spaces": [],
                                        "bases": [{
                                            "cform": "int",
                                            "loc": {
                                                "end": {"column": 7, "line": 33},
                                                "start": {"column": 3, "line": 33}
                                            },
                                            "raw": "int",
                                            "type": "Literal",
                                            "value": "int"
                                        }],
                                        "cform": "int ",
                                        "loc": {"end": {"column": 7, "line": 33}, "start": {"column": 3, "line": 33}},
                                        "qualifiers": [],
                                        "raw": "int numBColumns",
                                        "type": "TypeSpecification"
                                    },
                                    "loc": {"end": {"column": 7, "line": 33}, "start": {"column": 3, "line": 33}},
                                    "name": "numBColumns",
                                    "raw": "int numBColumns",
                                    "type": "Identifier"
                                },
                                "cform": "& numBColumns",
                                "loc": {"end": {"column": 68, "line": 43}, "start": {"column": 67, "line": 43}},
                                "operator": "&",
                                "prefix": true,
                                "raw": "&numBColumns",
                                "type": "UnaryExpression"
                            }],
                            "callee": {
                                "cform": "wbImport",
                                "loc": {"end": {"column": 79, "line": 43}, "start": {"column": 18, "line": 43}},
                                "name": "wbImport",
                                "raw": "wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns)",
                                "type": "Identifier"
                            },
                            "cform": "wbImport(\"input1\" /* String*/, & numBRows /* UnaryOperator*/, & numBColumns)",
                            "loc": {"end": {"column": 79, "line": 43}, "start": {"column": 18, "line": 43}},
                            "raw": "wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns)",
                            "type": "CallExpression"
                        },
                        "type": "AssignmentExpression"
                    }, {
                        "cform": "hostC = malloc(numARows * numBColumns * sizeof(float ))",
                        "left": {
                            "cform": "hostC",
                            "kind": {
                                "cform": "float *",
                                "loc": {"end": {"column": 10, "line": 26}, "start": {"column": 3, "line": 26}},
                                "raw": "float *hostC",
                                "type": "ReferenceType",
                                "value": {
                                    "address_spaces": [],
                                    "bases": [{
                                        "cform": "float",
                                        "loc": {"end": {"column": 10, "line": 26}, "start": {"column": 3, "line": 26}},
                                        "raw": "float",
                                        "type": "Literal",
                                        "value": "float"
                                    }],
                                    "cform": "float ",
                                    "loc": {"end": {"column": 10, "line": 26}, "start": {"column": 3, "line": 26}},
                                    "qualifiers": [],
                                    "raw": "float *hostC",
                                    "type": "TypeSpecification"
                                }
                            },
                            "loc": {"end": {"column": 10, "line": 26}, "start": {"column": 3, "line": 26}},
                            "name": "hostC",
                            "raw": "float *hostC",
                            "type": "Identifier"
                        },
                        "loc": {"end": {"column": 67, "line": 45}, "start": {"column": 3, "line": 45}},
                        "operator": "=",
                        "raw": "hostC = ( float * )malloc(numARows * numBColumns * sizeof(float))",
                        "right": {
                            "arguments": [{
                                "cform": "numARows * numBColumns * sizeof(float )",
                                "left": {
                                    "cform": "numARows * numBColumns",
                                    "left": {
                                        "cform": "numARows",
                                        "kind": {
                                            "address_spaces": [],
                                            "bases": [{
                                                "cform": "int",
                                                "loc": {
                                                    "end": {"column": 7, "line": 30},
                                                    "start": {"column": 3, "line": 30}
                                                },
                                                "raw": "int",
                                                "type": "Literal",
                                                "value": "int"
                                            }],
                                            "cform": "int ",
                                            "loc": {
                                                "end": {"column": 7, "line": 30},
                                                "start": {"column": 3, "line": 30}
                                            },
                                            "qualifiers": [],
                                            "raw": "int numARows",
                                            "type": "TypeSpecification"
                                        },
                                        "loc": {"end": {"column": 7, "line": 30}, "start": {"column": 3, "line": 30}},
                                        "name": "numARows",
                                        "raw": "int numARows",
                                        "type": "Identifier"
                                    },
                                    "loc": {"end": {"column": 40, "line": 45}, "start": {"column": 29, "line": 45}},
                                    "operator": "*",
                                    "raw": "numARows * numBColumns",
                                    "right": {
                                        "cform": "numBColumns",
                                        "kind": {
                                            "address_spaces": [],
                                            "bases": [{
                                                "cform": "int",
                                                "loc": {
                                                    "end": {"column": 7, "line": 33},
                                                    "start": {"column": 3, "line": 33}
                                                },
                                                "raw": "int",
                                                "type": "Literal",
                                                "value": "int"
                                            }],
                                            "cform": "int ",
                                            "loc": {
                                                "end": {"column": 7, "line": 33},
                                                "start": {"column": 3, "line": 33}
                                            },
                                            "qualifiers": [],
                                            "raw": "int numBColumns",
                                            "type": "TypeSpecification"
                                        },
                                        "loc": {"end": {"column": 7, "line": 33}, "start": {"column": 3, "line": 33}},
                                        "name": "numBColumns",
                                        "raw": "int numBColumns",
                                        "type": "Identifier"
                                    },
                                    "type": "BinaryExpression"
                                },
                                "loc": {"end": {"column": 66, "line": 45}, "start": {"column": 29, "line": 45}},
                                "operator": "*",
                                "raw": "numARows * numBColumns * sizeof(float)",
                                "right": {
                                    "arguments": [{
                                        "address_spaces": [],
                                        "bases": [{
                                            "cform": "float",
                                            "loc": {
                                                "end": {"column": 66, "line": 45},
                                                "start": {"column": 54, "line": 45}
                                            },
                                            "raw": "float",
                                            "type": "Literal",
                                            "value": "float"
                                        }],
                                        "cform": "float ",
                                        "loc": {"end": {"column": 66, "line": 45}, "start": {"column": 54, "line": 45}},
                                        "qualifiers": [],
                                        "raw": "sizeof(float)",
                                        "type": "TypeSpecification"
                                    }],
                                    "callee": {
                                        "cform": "sizeof",
                                        "loc": {"end": {"column": 66, "line": 45}, "start": {"column": 54, "line": 45}},
                                        "name": "sizeof",
                                        "raw": "sizeof(float)",
                                        "type": "Identifier"
                                    },
                                    "cform": "sizeof(float )",
                                    "loc": {"end": {"column": 66, "line": 45}, "start": {"column": 54, "line": 45}},
                                    "raw": "sizeof(float)",
                                    "type": "CallExpression"
                                },
                                "type": "BinaryExpression"
                            }],
                            "callee": {
                                "cform": "malloc",
                                "loc": {"end": {"column": 67, "line": 45}, "start": {"column": 22, "line": 45}},
                                "name": "malloc",
                                "raw": "malloc(numARows * numBColumns * sizeof(float))",
                                "type": "Identifier"
                            },
                            "cform": "malloc(numARows * numBColumns * sizeof(float ))",
                            "loc": {"end": {"column": 67, "line": 45}, "start": {"column": 22, "line": 45}},
                            "raw": "malloc(numARows * numBColumns * sizeof(float))",
                            "type": "CallExpression"
                        },
                        "type": "AssignmentExpression"
                    }, {
                        "arguments": [{
                            "cform": "\"Generic\"",
                            "loc": {"end": {"column": 3, "line": 46}, "start": {"column": 3, "line": 46}},
                            "raw": "Generic",
                            "type": "StringLiteral",
                            "value": "\"Generic\""
                        }, {
                            "cform": "\"Importing data and creating memory on host\"",
                            "loc": {"end": {"column": 3, "line": 46}, "start": {"column": 3, "line": 46}},
                            "raw": "Importing data and creating memory on host",
                            "type": "StringLiteral",
                            "value": "\"Importing data and creating memory on host\""
                        }],
                        "callee": {
                            "cform": "wbTime_stop",
                            "loc": {"end": {"column": 3, "line": 46}, "start": {"column": 3, "line": 46}},
                            "name": "wbTime_stop",
                            "raw": "wbTime_stop(#kind, __VA_ARGS__)",
                            "type": "Identifier"
                        },
                        "cform": "wbTime_stop(\"Generic\" /* String*/, \"Importing data and creating memory on host\")",
                        "loc": {"end": {"column": 3, "line": 46}, "start": {"column": 3, "line": 46}},
                        "raw": "wbTime_stop(#kind, __VA_ARGS__)",
                        "type": "CallExpression"
                    }, {}, {
                        "cform": "numCRows = numARows",
                        "left": {
                            "cform": "numCRows",
                            "kind": {
                                "address_spaces": [],
                                "bases": [{
                                    "cform": "int",
                                    "loc": {"end": {"column": 7, "line": 34}, "start": {"column": 3, "line": 34}},
                                    "raw": "int",
                                    "type": "Literal",
                                    "value": "int"
                                }],
                                "cform": "int ",
                                "loc": {"end": {"column": 7, "line": 34}, "start": {"column": 3, "line": 34}},
                                "qualifiers": [],
                                "raw": "int numCRows",
                                "type": "TypeSpecification"
                            },
                            "loc": {"end": {"column": 7, "line": 34}, "start": {"column": 3, "line": 34}},
                            "name": "numCRows",
                            "raw": "int numCRows",
                            "type": "Identifier"
                        },
                        "loc": {"end": {"column": 14, "line": 48}, "start": {"column": 3, "line": 48}},
                        "operator": "=",
                        "raw": "numCRows = numARows",
                        "right": {
                            "cform": "numARows",
                            "kind": {
                                "address_spaces": [],
                                "bases": [{
                                    "cform": "int",
                                    "loc": {"end": {"column": 7, "line": 30}, "start": {"column": 3, "line": 30}},
                                    "raw": "int",
                                    "type": "Literal",
                                    "value": "int"
                                }],
                                "cform": "int ",
                                "loc": {"end": {"column": 7, "line": 30}, "start": {"column": 3, "line": 30}},
                                "qualifiers": [],
                                "raw": "int numARows",
                                "type": "TypeSpecification"
                            },
                            "loc": {"end": {"column": 7, "line": 30}, "start": {"column": 3, "line": 30}},
                            "name": "numARows",
                            "raw": "int numARows",
                            "type": "Identifier"
                        },
                        "type": "AssignmentExpression"
                    }, {
                        "cform": "numCColumns = numBColumns",
                        "left": {
                            "cform": "numCColumns",
                            "kind": {
                                "address_spaces": [],
                                "bases": [{
                                    "cform": "int",
                                    "loc": {"end": {"column": 7, "line": 35}, "start": {"column": 3, "line": 35}},
                                    "raw": "int",
                                    "type": "Literal",
                                    "value": "int"
                                }],
                                "cform": "int ",
                                "loc": {"end": {"column": 7, "line": 35}, "start": {"column": 3, "line": 35}},
                                "qualifiers": [],
                                "raw": "int numCColumns",
                                "type": "TypeSpecification"
                            },
                            "loc": {"end": {"column": 7, "line": 35}, "start": {"column": 3, "line": 35}},
                            "name": "numCColumns",
                            "raw": "int numCColumns",
                            "type": "Identifier"
                        },
                        "loc": {"end": {"column": 17, "line": 49}, "start": {"column": 3, "line": 49}},
                        "operator": "=",
                        "raw": "numCColumns = numBColumns",
                        "right": {
                            "cform": "numBColumns",
                            "kind": {
                                "address_spaces": [],
                                "bases": [{
                                    "cform": "int",
                                    "loc": {"end": {"column": 7, "line": 33}, "start": {"column": 3, "line": 33}},
                                    "raw": "int",
                                    "type": "Literal",
                                    "value": "int"
                                }],
                                "cform": "int ",
                                "loc": {"end": {"column": 7, "line": 33}, "start": {"column": 3, "line": 33}},
                                "qualifiers": [],
                                "raw": "int numBColumns",
                                "type": "TypeSpecification"
                            },
                            "loc": {"end": {"column": 7, "line": 33}, "start": {"column": 3, "line": 33}},
                            "name": "numBColumns",
                            "raw": "int numBColumns",
                            "type": "Identifier"
                        },
                        "type": "AssignmentExpression"
                    }, {
                        "arguments": [{
                            "cform": "\"TRACE\"",
                            "loc": {"end": {"column": 3, "line": 51}, "start": {"column": 3, "line": 51}},
                            "raw": "TRACE",
                            "type": "StringLiteral",
                            "value": "\"TRACE\""
                        }, {
                            "cform": "\"The dimensions of A are \"",
                            "loc": {"end": {"column": 3, "line": 51}, "start": {"column": 3, "line": 51}},
                            "raw": "The dimensions of A are ",
                            "type": "StringLiteral",
                            "value": "\"The dimensions of A are \""
                        }, {
                            "cform": "numARows",
                            "kind": {
                                "address_spaces": [],
                                "bases": [{
                                    "cform": "int",
                                    "loc": {"end": {"column": 7, "line": 30}, "start": {"column": 3, "line": 30}},
                                    "raw": "int",
                                    "type": "Literal",
                                    "value": "int"
                                }],
                                "cform": "int ",
                                "loc": {"end": {"column": 7, "line": 30}, "start": {"column": 3, "line": 30}},
                                "qualifiers": [],
                                "raw": "int numARows",
                                "type": "TypeSpecification"
                            },
                            "loc": {"end": {"column": 7, "line": 30}, "start": {"column": 3, "line": 30}},
                            "name": "numARows",
                            "raw": "int numARows",
                            "type": "Identifier"
                        }, {
                            "cform": "\" x \"",
                            "loc": {"end": {"column": 3, "line": 51}, "start": {"column": 3, "line": 51}},
                            "raw": " x ",
                            "type": "StringLiteral",
                            "value": "\" x \""
                        }, {
                            "cform": "numAColumns",
                            "kind": {
                                "address_spaces": [],
                                "bases": [{
                                    "cform": "int",
                                    "loc": {"end": {"column": 7, "line": 31}, "start": {"column": 3, "line": 31}},
                                    "raw": "int",
                                    "type": "Literal",
                                    "value": "int"
                                }],
                                "cform": "int ",
                                "loc": {"end": {"column": 7, "line": 31}, "start": {"column": 3, "line": 31}},
                                "qualifiers": [],
                                "raw": "int numAColumns",
                                "type": "TypeSpecification"
                            },
                            "loc": {"end": {"column": 7, "line": 31}, "start": {"column": 3, "line": 31}},
                            "name": "numAColumns",
                            "raw": "int numAColumns",
                            "type": "Identifier"
                        }],
                        "callee": {
                            "cform": "wbLog",
                            "loc": {"end": {"column": 3, "line": 51}, "start": {"column": 3, "line": 51}},
                            "name": "wbLog",
                            "raw": "wbLog(#level, __VA_ARGS__)",
                            "type": "Identifier"
                        },
                        "cform": "wbLog(\"TRACE\" /* String*/, \"The dimensions of A are \" /* String*/, numARows /* Identifier*/, \" x \" /* String*/, numAColumns)",
                        "loc": {"end": {"column": 3, "line": 51}, "start": {"column": 3, "line": 51}},
                        "raw": "wbLog(#level, __VA_ARGS__)",
                        "type": "CallExpression"
                    }, {
                        "arguments": [{
                            "cform": "\"TRACE\"",
                            "loc": {"end": {"column": 3, "line": 52}, "start": {"column": 3, "line": 52}},
                            "raw": "TRACE",
                            "type": "StringLiteral",
                            "value": "\"TRACE\""
                        }, {
                            "cform": "\"The dimensions of B are \"",
                            "loc": {"end": {"column": 3, "line": 52}, "start": {"column": 3, "line": 52}},
                            "raw": "The dimensions of B are ",
                            "type": "StringLiteral",
                            "value": "\"The dimensions of B are \""
                        }, {
                            "cform": "numBRows",
                            "kind": {
                                "address_spaces": [],
                                "bases": [{
                                    "cform": "int",
                                    "loc": {"end": {"column": 7, "line": 32}, "start": {"column": 3, "line": 32}},
                                    "raw": "int",
                                    "type": "Literal",
                                    "value": "int"
                                }],
                                "cform": "int ",
                                "loc": {"end": {"column": 7, "line": 32}, "start": {"column": 3, "line": 32}},
                                "qualifiers": [],
                                "raw": "int numBRows",
                                "type": "TypeSpecification"
                            },
                            "loc": {"end": {"column": 7, "line": 32}, "start": {"column": 3, "line": 32}},
                            "name": "numBRows",
                            "raw": "int numBRows",
                            "type": "Identifier"
                        }, {
                            "cform": "\" x \"",
                            "loc": {"end": {"column": 3, "line": 52}, "start": {"column": 3, "line": 52}},
                            "raw": " x ",
                            "type": "StringLiteral",
                            "value": "\" x \""
                        }, {
                            "cform": "numBColumns",
                            "kind": {
                                "address_spaces": [],
                                "bases": [{
                                    "cform": "int",
                                    "loc": {"end": {"column": 7, "line": 33}, "start": {"column": 3, "line": 33}},
                                    "raw": "int",
                                    "type": "Literal",
                                    "value": "int"
                                }],
                                "cform": "int ",
                                "loc": {"end": {"column": 7, "line": 33}, "start": {"column": 3, "line": 33}},
                                "qualifiers": [],
                                "raw": "int numBColumns",
                                "type": "TypeSpecification"
                            },
                            "loc": {"end": {"column": 7, "line": 33}, "start": {"column": 3, "line": 33}},
                            "name": "numBColumns",
                            "raw": "int numBColumns",
                            "type": "Identifier"
                        }],
                        "callee": {
                            "cform": "wbLog",
                            "loc": {"end": {"column": 3, "line": 52}, "start": {"column": 3, "line": 52}},
                            "name": "wbLog",
                            "raw": "wbLog(#level, __VA_ARGS__)",
                            "type": "Identifier"
                        },
                        "cform": "wbLog(\"TRACE\" /* String*/, \"The dimensions of B are \" /* String*/, numBRows /* Identifier*/, \" x \" /* String*/, numBColumns)",
                        "loc": {"end": {"column": 3, "line": 52}, "start": {"column": 3, "line": 52}},
                        "raw": "wbLog(#level, __VA_ARGS__)",
                        "type": "CallExpression"
                    }, {
                        "arguments": [{
                            "cform": "\"TRACE\"",
                            "loc": {"end": {"column": 3, "line": 53}, "start": {"column": 3, "line": 53}},
                            "raw": "TRACE",
                            "type": "StringLiteral",
                            "value": "\"TRACE\""
                        }, {
                            "cform": "\"The dimensions of C are \"",
                            "loc": {"end": {"column": 3, "line": 53}, "start": {"column": 3, "line": 53}},
                            "raw": "The dimensions of C are ",
                            "type": "StringLiteral",
                            "value": "\"The dimensions of C are \""
                        }, {
                            "cform": "numCRows",
                            "kind": {
                                "address_spaces": [],
                                "bases": [{
                                    "cform": "int",
                                    "loc": {"end": {"column": 7, "line": 34}, "start": {"column": 3, "line": 34}},
                                    "raw": "int",
                                    "type": "Literal",
                                    "value": "int"
                                }],
                                "cform": "int ",
                                "loc": {"end": {"column": 7, "line": 34}, "start": {"column": 3, "line": 34}},
                                "qualifiers": [],
                                "raw": "int numCRows",
                                "type": "TypeSpecification"
                            },
                            "loc": {"end": {"column": 7, "line": 34}, "start": {"column": 3, "line": 34}},
                            "name": "numCRows",
                            "raw": "int numCRows",
                            "type": "Identifier"
                        }, {
                            "cform": "\" x \"",
                            "loc": {"end": {"column": 3, "line": 53}, "start": {"column": 3, "line": 53}},
                            "raw": " x ",
                            "type": "StringLiteral",
                            "value": "\" x \""
                        }, {
                            "cform": "numCColumns",
                            "kind": {
                                "address_spaces": [],
                                "bases": [{
                                    "cform": "int",
                                    "loc": {"end": {"column": 7, "line": 35}, "start": {"column": 3, "line": 35}},
                                    "raw": "int",
                                    "type": "Literal",
                                    "value": "int"
                                }],
                                "cform": "int ",
                                "loc": {"end": {"column": 7, "line": 35}, "start": {"column": 3, "line": 35}},
                                "qualifiers": [],
                                "raw": "int numCColumns",
                                "type": "TypeSpecification"
                            },
                            "loc": {"end": {"column": 7, "line": 35}, "start": {"column": 3, "line": 35}},
                            "name": "numCColumns",
                            "raw": "int numCColumns",
                            "type": "Identifier"
                        }],
                        "callee": {
                            "cform": "wbLog",
                            "loc": {"end": {"column": 3, "line": 53}, "start": {"column": 3, "line": 53}},
                            "name": "wbLog",
                            "raw": "wbLog(#level, __VA_ARGS__)",
                            "type": "Identifier"
                        },
                        "cform": "wbLog(\"TRACE\" /* String*/, \"The dimensions of C are \" /* String*/, numCRows /* Identifier*/, \" x \" /* String*/, numCColumns)",
                        "loc": {"end": {"column": 3, "line": 53}, "start": {"column": 3, "line": 53}},
                        "raw": "wbLog(#level, __VA_ARGS__)",
                        "type": "CallExpression"
                    }, {
                        "arguments": [{
                            "cform": "\"GPU\"",
                            "loc": {"end": {"column": 3, "line": 55}, "start": {"column": 3, "line": 55}},
                            "raw": "GPU",
                            "type": "StringLiteral",
                            "value": "\"GPU\""
                        }, {
                            "cform": "\"Allocating GPU memory.\"",
                            "loc": {"end": {"column": 3, "line": 55}, "start": {"column": 3, "line": 55}},
                            "raw": "Allocating GPU memory.",
                            "type": "StringLiteral",
                            "value": "\"Allocating GPU memory.\""
                        }],
                        "callee": {
                            "cform": "wbTime_start",
                            "loc": {"end": {"column": 3, "line": 55}, "start": {"column": 3, "line": 55}},
                            "name": "wbTime_start",
                            "raw": "wbTime_start(#kind, __VA_ARGS__)",
                            "type": "Identifier"
                        },
                        "cform": "wbTime_start(\"GPU\" /* String*/, \"Allocating GPU memory.\")",
                        "loc": {"end": {"column": 3, "line": 55}, "start": {"column": 3, "line": 55}},
                        "raw": "wbTime_start(#kind, __VA_ARGS__)",
                        "type": "CallExpression"
                    }, {}, {}, {}, {}, {
                        "arguments": [{
                            "cform": "\"GPU\"",
                            "loc": {"end": {"column": 3, "line": 63}, "start": {"column": 3, "line": 63}},
                            "raw": "GPU",
                            "type": "StringLiteral",
                            "value": "\"GPU\""
                        }, {
                            "cform": "\"Allocating GPU memory.\"",
                            "loc": {"end": {"column": 3, "line": 63}, "start": {"column": 3, "line": 63}},
                            "raw": "Allocating GPU memory.",
                            "type": "StringLiteral",
                            "value": "\"Allocating GPU memory.\""
                        }],
                        "callee": {
                            "cform": "wbTime_stop",
                            "loc": {"end": {"column": 3, "line": 63}, "start": {"column": 3, "line": 63}},
                            "name": "wbTime_stop",
                            "raw": "wbTime_stop(#kind, __VA_ARGS__)",
                            "type": "Identifier"
                        },
                        "cform": "wbTime_stop(\"GPU\" /* String*/, \"Allocating GPU memory.\")",
                        "loc": {"end": {"column": 3, "line": 63}, "start": {"column": 3, "line": 63}},
                        "raw": "wbTime_stop(#kind, __VA_ARGS__)",
                        "type": "CallExpression"
                    }, {}, {
                        "arguments": [{
                            "cform": "\"GPU\"",
                            "loc": {"end": {"column": 3, "line": 65}, "start": {"column": 3, "line": 65}},
                            "raw": "GPU",
                            "type": "StringLiteral",
                            "value": "\"GPU\""
                        }, {
                            "cform": "\"Copying input memory to the GPU.\"",
                            "loc": {"end": {"column": 3, "line": 65}, "start": {"column": 3, "line": 65}},
                            "raw": "Copying input memory to the GPU.",
                            "type": "StringLiteral",
                            "value": "\"Copying input memory to the GPU.\""
                        }],
                        "callee": {
                            "cform": "wbTime_start",
                            "loc": {"end": {"column": 3, "line": 65}, "start": {"column": 3, "line": 65}},
                            "name": "wbTime_start",
                            "raw": "wbTime_start(#kind, __VA_ARGS__)",
                            "type": "Identifier"
                        },
                        "cform": "wbTime_start(\"GPU\" /* String*/, \"Copying input memory to the GPU.\")",
                        "loc": {"end": {"column": 3, "line": 65}, "start": {"column": 3, "line": 65}},
                        "raw": "wbTime_start(#kind, __VA_ARGS__)",
                        "type": "CallExpression"
                    }, {}, {}, {}, {
                        "arguments": [{
                            "cform": "\"GPU\"",
                            "loc": {"end": {"column": 3, "line": 71}, "start": {"column": 3, "line": 71}},
                            "raw": "GPU",
                            "type": "StringLiteral",
                            "value": "\"GPU\""
                        }, {
                            "cform": "\"Copying input memory to the GPU.\"",
                            "loc": {"end": {"column": 3, "line": 71}, "start": {"column": 3, "line": 71}},
                            "raw": "Copying input memory to the GPU.",
                            "type": "StringLiteral",
                            "value": "\"Copying input memory to the GPU.\""
                        }],
                        "callee": {
                            "cform": "wbTime_stop",
                            "loc": {"end": {"column": 3, "line": 71}, "start": {"column": 3, "line": 71}},
                            "name": "wbTime_stop",
                            "raw": "wbTime_stop(#kind, __VA_ARGS__)",
                            "type": "Identifier"
                        },
                        "cform": "wbTime_stop(\"GPU\" /* String*/, \"Copying input memory to the GPU.\")",
                        "loc": {"end": {"column": 3, "line": 71}, "start": {"column": 3, "line": 71}},
                        "raw": "wbTime_stop(#kind, __VA_ARGS__)",
                        "type": "CallExpression"
                    }, {}, {
                        "cform": "struct dim3  blockDim = {16 /* Integer32*/, 16}",
                        "declarations": [{
                            "cform": "struct dim3  blockDim = {16 /* Integer32*/, 16}",
                            "id": {
                                "cform": "blockDim",
                                "loc": {"end": {"column": 23, "line": 74}, "start": {"column": 3, "line": 74}},
                                "name": "blockDim",
                                "raw": "dim3 blockDim(16, 16)",
                                "type": "Identifier"
                            },
                            "init": {
                                "cform": "{16 /* Integer32*/, 16}",
                                "elements": [{
                                    "cform": "16",
                                    "loc": {"end": {"column": 17, "line": 74}, "start": {"column": 17, "line": 74}},
                                    "raw": "16",
                                    "type": "Integer32Literal",
                                    "value": 16
                                }, {
                                    "cform": "16",
                                    "loc": {"end": {"column": 21, "line": 74}, "start": {"column": 21, "line": 74}},
                                    "raw": "16",
                                    "type": "Integer32Literal",
                                    "value": 16
                                }],
                                "loc": {"end": {"column": 23, "line": 74}, "start": {"column": 8, "line": 74}},
                                "raw": "blockDim(16, 16)",
                                "type": "ArrayExpression"
                            },
                            "loc": {"end": {"column": 23, "line": 74}, "start": {"column": 3, "line": 74}},
                            "raw": "dim3 blockDim(16, 16)",
                            "type": "VariableDeclarator"
                        }],
                        "loc": {"end": {"column": 23, "line": 74}, "start": {"column": 3, "line": 74}},
                        "raw": "dim3 blockDim(16, 16)",
                        "type": "VariableDeclaration"
                    }, {
                        "cform": "struct dim3  gridDim = {ceil((numAColumns) / blockDim.x) /* Call*/, ceil((numBRows) / blockDim.y)}",
                        "declarations": [{
                            "cform": "struct dim3  gridDim = {ceil((numAColumns) / blockDim.x) /* Call*/, ceil((numBRows) / blockDim.y)}",
                            "id": {
                                "cform": "gridDim",
                                "loc": {"end": {"column": 54, "line": 76}, "start": {"column": 3, "line": 75}},
                                "name": "gridDim",
                                "raw": "lockDim.y))...",
                                "type": "Identifier"
                            },
                            "init": {
                                "cform": "{ceil((numAColumns) / blockDim.x) /* Call*/, ceil((numBRows) / blockDim.y)}",
                                "elements": [{
                                    "arguments": [{
                                        "cform": "(numAColumns) / blockDim.x",
                                        "left": {
                                            "cform": "numAColumns",
                                            "kind": {
                                                "address_spaces": [],
                                                "bases": [{
                                                    "cform": "int",
                                                    "loc": {
                                                        "end": {"column": 7, "line": 31},
                                                        "start": {"column": 3, "line": 31}
                                                    },
                                                    "raw": "int",
                                                    "type": "Literal",
                                                    "value": "int"
                                                }],
                                                "cform": "int ",
                                                "loc": {
                                                    "end": {"column": 7, "line": 31},
                                                    "start": {"column": 3, "line": 31}
                                                },
                                                "qualifiers": [],
                                                "raw": "int numAColumns",
                                                "type": "TypeSpecification"
                                            },
                                            "loc": {
                                                "end": {"column": 7, "line": 31},
                                                "start": {"column": 3, "line": 31}
                                            },
                                            "name": "numAColumns",
                                            "raw": "int numAColumns",
                                            "type": "Identifier"
                                        },
                                        "loc": {"end": {"column": 55, "line": 75}, "start": {"column": 21, "line": 75}},
                                        "operator": "/",
                                        "raw": "(( float )numAColumns) / blockDim.x",
                                        "right": {
                                            "cform": "blockDim.x",
                                            "left": {
                                                "cform": "blockDim",
                                                "kind": {
                                                    "address_spaces": [],
                                                    "bases": [{
                                                        "cform": "struct dim3",
                                                        "loc": {
                                                            "end": {"column": 23, "line": 74},
                                                            "start": {"column": 3, "line": 74}
                                                        },
                                                        "raw": "struct dim3",
                                                        "type": "Literal",
                                                        "value": "struct dim3"
                                                    }],
                                                    "cform": "struct dim3 ",
                                                    "loc": {
                                                        "end": {"column": 23, "line": 74},
                                                        "start": {"column": 3, "line": 74}
                                                    },
                                                    "qualifiers": [],
                                                    "raw": "dim3 blockDim(16, 16)",
                                                    "type": "TypeSpecification"
                                                },
                                                "loc": {
                                                    "end": {"column": 23, "line": 74},
                                                    "start": {"column": 3, "line": 74}
                                                },
                                                "name": "blockDim",
                                                "raw": "dim3 blockDim(16, 16)",
                                                "type": "Identifier"
                                            },
                                            "loc": {
                                                "end": {"column": 55, "line": 75},
                                                "start": {"column": 46, "line": 75}
                                            },
                                            "operator": ".",
                                            "raw": "blockDim.x",
                                            "right": {
                                                "cform": "x",
                                                "kind": {
                                                    "address_spaces": [],
                                                    "bases": [{
                                                        "cform": "unsigned int",
                                                        "loc": {
                                                            "end": {"column": 16, "line": 16},
                                                            "start": {"column": 3, "line": 16}
                                                        },
                                                        "raw": "unsigned int",
                                                        "type": "Literal",
                                                        "value": "unsigned int"
                                                    }],
                                                    "cform": "unsigned int ",
                                                    "loc": {
                                                        "end": {"column": 16, "line": 16},
                                                        "start": {"column": 3, "line": 16}
                                                    },
                                                    "qualifiers": [],
                                                    "raw": "unsigned int x",
                                                    "type": "TypeSpecification"
                                                },
                                                "loc": {
                                                    "end": {"column": 16, "line": 16},
                                                    "start": {"column": 3, "line": 16}
                                                },
                                                "name": "x",
                                                "raw": "unsigned int x",
                                                "type": "Identifier"
                                            },
                                            "type": "MemberExpression"
                                        },
                                        "type": "BinaryExpression"
                                    }],
                                    "callee": {
                                        "cform": "ceil",
                                        "loc": {"end": {"column": 56, "line": 75}, "start": {"column": 16, "line": 75}},
                                        "name": "ceil",
                                        "raw": "ceil((( float )numAColumns) / blockDim.x)",
                                        "type": "Identifier"
                                    },
                                    "cform": "ceil((numAColumns) / blockDim.x)",
                                    "loc": {"end": {"column": 56, "line": 75}, "start": {"column": 16, "line": 75}},
                                    "raw": "ceil((( float )numAColumns) / blockDim.x)",
                                    "type": "CallExpression"
                                }, {
                                    "arguments": [{
                                        "cform": "(numBRows) / blockDim.y",
                                        "left": {
                                            "cform": "numBRows",
                                            "kind": {
                                                "address_spaces": [],
                                                "bases": [{
                                                    "cform": "int",
                                                    "loc": {
                                                        "end": {"column": 7, "line": 32},
                                                        "start": {"column": 3, "line": 32}
                                                    },
                                                    "raw": "int",
                                                    "type": "Literal",
                                                    "value": "int"
                                                }],
                                                "cform": "int ",
                                                "loc": {
                                                    "end": {"column": 7, "line": 32},
                                                    "start": {"column": 3, "line": 32}
                                                },
                                                "qualifiers": [],
                                                "raw": "int numBRows",
                                                "type": "TypeSpecification"
                                            },
                                            "loc": {
                                                "end": {"column": 7, "line": 32},
                                                "start": {"column": 3, "line": 32}
                                            },
                                            "name": "numBRows",
                                            "raw": "int numBRows",
                                            "type": "Identifier"
                                        },
                                        "loc": {"end": {"column": 52, "line": 76}, "start": {"column": 21, "line": 76}},
                                        "operator": "/",
                                        "raw": "(( float )numBRows) / blockDim.y",
                                        "right": {
                                            "cform": "blockDim.y",
                                            "left": {
                                                "cform": "blockDim",
                                                "kind": {
                                                    "address_spaces": [],
                                                    "bases": [{
                                                        "cform": "struct dim3",
                                                        "loc": {
                                                            "end": {"column": 23, "line": 74},
                                                            "start": {"column": 3, "line": 74}
                                                        },
                                                        "raw": "struct dim3",
                                                        "type": "Literal",
                                                        "value": "struct dim3"
                                                    }],
                                                    "cform": "struct dim3 ",
                                                    "loc": {
                                                        "end": {"column": 23, "line": 74},
                                                        "start": {"column": 3, "line": 74}
                                                    },
                                                    "qualifiers": [],
                                                    "raw": "dim3 blockDim(16, 16)",
                                                    "type": "TypeSpecification"
                                                },
                                                "loc": {
                                                    "end": {"column": 23, "line": 74},
                                                    "start": {"column": 3, "line": 74}
                                                },
                                                "name": "blockDim",
                                                "raw": "dim3 blockDim(16, 16)",
                                                "type": "Identifier"
                                            },
                                            "loc": {
                                                "end": {"column": 52, "line": 76},
                                                "start": {"column": 43, "line": 76}
                                            },
                                            "operator": ".",
                                            "raw": "blockDim.y",
                                            "right": {
                                                "cform": "y",
                                                "kind": {
                                                    "address_spaces": [],
                                                    "bases": [{
                                                        "cform": "unsigned int",
                                                        "loc": {
                                                            "end": {"column": 19, "line": 16},
                                                            "start": {"column": 3, "line": 16}
                                                        },
                                                        "raw": "unsigned int",
                                                        "type": "Literal",
                                                        "value": "unsigned int"
                                                    }],
                                                    "cform": "unsigned int ",
                                                    "loc": {
                                                        "end": {"column": 19, "line": 16},
                                                        "start": {"column": 3, "line": 16}
                                                    },
                                                    "qualifiers": [],
                                                    "raw": "unsigned int x, y",
                                                    "type": "TypeSpecification"
                                                },
                                                "loc": {
                                                    "end": {"column": 19, "line": 16},
                                                    "start": {"column": 3, "line": 16}
                                                },
                                                "name": "y",
                                                "raw": "unsigned int x, y",
                                                "type": "Identifier"
                                            },
                                            "type": "MemberExpression"
                                        },
                                        "type": "BinaryExpression"
                                    }],
                                    "callee": {
                                        "cform": "ceil",
                                        "loc": {"end": {"column": 53, "line": 76}, "start": {"column": 16, "line": 76}},
                                        "name": "ceil",
                                        "raw": "ceil((( float )numBRows) / blockDim.y)",
                                        "type": "Identifier"
                                    },
                                    "cform": "ceil((numBRows) / blockDim.y)",
                                    "loc": {"end": {"column": 53, "line": 76}, "start": {"column": 16, "line": 76}},
                                    "raw": "ceil((( float )numBRows) / blockDim.y)",
                                    "type": "CallExpression"
                                }],
                                "loc": {"end": {"column": 54, "line": 76}, "start": {"column": 8, "line": 75}},
                                "raw": "im.y))...",
                                "type": "ArrayExpression"
                            },
                            "loc": {"end": {"column": 54, "line": 76}, "start": {"column": 3, "line": 75}},
                            "raw": "lockDim.y))...",
                            "type": "VariableDeclarator"
                        }],
                        "loc": {"end": {"column": 54, "line": 76}, "start": {"column": 3, "line": 75}},
                        "raw": "lockDim.y))...",
                        "type": "VariableDeclaration"
                    }, {
                        "arguments": [{
                            "cform": "\"TRACE\"",
                            "loc": {"end": {"column": 3, "line": 78}, "start": {"column": 3, "line": 78}},
                            "raw": "TRACE",
                            "type": "StringLiteral",
                            "value": "\"TRACE\""
                        }, {
                            "cform": "\"The block dimensions are \"",
                            "loc": {"end": {"column": 3, "line": 78}, "start": {"column": 3, "line": 78}},
                            "raw": "The block dimensions are ",
                            "type": "StringLiteral",
                            "value": "\"The block dimensions are \""
                        }, {
                            "cform": "blockDim.x",
                            "left": {
                                "cform": "blockDim",
                                "kind": {
                                    "address_spaces": [],
                                    "bases": [{
                                        "cform": "struct dim3",
                                        "loc": {"end": {"column": 23, "line": 74}, "start": {"column": 3, "line": 74}},
                                        "raw": "struct dim3",
                                        "type": "Literal",
                                        "value": "struct dim3"
                                    }],
                                    "cform": "struct dim3 ",
                                    "loc": {"end": {"column": 23, "line": 74}, "start": {"column": 3, "line": 74}},
                                    "qualifiers": [],
                                    "raw": "dim3 blockDim(16, 16)",
                                    "type": "TypeSpecification"
                                },
                                "loc": {"end": {"column": 23, "line": 74}, "start": {"column": 3, "line": 74}},
                                "name": "blockDim",
                                "raw": "dim3 blockDim(16, 16)",
                                "type": "Identifier"
                            },
                            "loc": {"end": {"column": 3, "line": 78}, "start": {"column": 3, "line": 78}},
                            "operator": ".",
                            "raw": "blockDim.x",
                            "right": {
                                "cform": "x",
                                "kind": {
                                    "address_spaces": [],
                                    "bases": [{
                                        "cform": "unsigned int",
                                        "loc": {"end": {"column": 16, "line": 16}, "start": {"column": 3, "line": 16}},
                                        "raw": "unsigned int",
                                        "type": "Literal",
                                        "value": "unsigned int"
                                    }],
                                    "cform": "unsigned int ",
                                    "loc": {"end": {"column": 16, "line": 16}, "start": {"column": 3, "line": 16}},
                                    "qualifiers": [],
                                    "raw": "unsigned int x",
                                    "type": "TypeSpecification"
                                },
                                "loc": {"end": {"column": 16, "line": 16}, "start": {"column": 3, "line": 16}},
                                "name": "x",
                                "raw": "unsigned int x",
                                "type": "Identifier"
                            },
                            "type": "MemberExpression"
                        }, {
                            "cform": "\" x \"",
                            "loc": {"end": {"column": 3, "line": 78}, "start": {"column": 3, "line": 78}},
                            "raw": " x ",
                            "type": "StringLiteral",
                            "value": "\" x \""
                        }, {
                            "cform": "blockDim.y",
                            "left": {
                                "cform": "blockDim",
                                "kind": {
                                    "address_spaces": [],
                                    "bases": [{
                                        "cform": "struct dim3",
                                        "loc": {"end": {"column": 23, "line": 74}, "start": {"column": 3, "line": 74}},
                                        "raw": "struct dim3",
                                        "type": "Literal",
                                        "value": "struct dim3"
                                    }],
                                    "cform": "struct dim3 ",
                                    "loc": {"end": {"column": 23, "line": 74}, "start": {"column": 3, "line": 74}},
                                    "qualifiers": [],
                                    "raw": "dim3 blockDim(16, 16)",
                                    "type": "TypeSpecification"
                                },
                                "loc": {"end": {"column": 23, "line": 74}, "start": {"column": 3, "line": 74}},
                                "name": "blockDim",
                                "raw": "dim3 blockDim(16, 16)",
                                "type": "Identifier"
                            },
                            "loc": {"end": {"column": 3, "line": 78}, "start": {"column": 3, "line": 78}},
                            "operator": ".",
                            "raw": "blockDim.y",
                            "right": {
                                "cform": "y",
                                "kind": {
                                    "address_spaces": [],
                                    "bases": [{
                                        "cform": "unsigned int",
                                        "loc": {"end": {"column": 19, "line": 16}, "start": {"column": 3, "line": 16}},
                                        "raw": "unsigned int",
                                        "type": "Literal",
                                        "value": "unsigned int"
                                    }],
                                    "cform": "unsigned int ",
                                    "loc": {"end": {"column": 19, "line": 16}, "start": {"column": 3, "line": 16}},
                                    "qualifiers": [],
                                    "raw": "unsigned int x, y",
                                    "type": "TypeSpecification"
                                },
                                "loc": {"end": {"column": 19, "line": 16}, "start": {"column": 3, "line": 16}},
                                "name": "y",
                                "raw": "unsigned int x, y",
                                "type": "Identifier"
                            },
                            "type": "MemberExpression"
                        }],
                        "callee": {
                            "cform": "wbLog",
                            "loc": {"end": {"column": 3, "line": 78}, "start": {"column": 3, "line": 78}},
                            "name": "wbLog",
                            "raw": "wbLog(#level, __VA_ARGS__)",
                            "type": "Identifier"
                        },
                        "cform": "wbLog(\"TRACE\" /* String*/, \"The block dimensions are \" /* String*/, blockDim.x /* Member*/, \" x \" /* String*/, blockDim.y)",
                        "loc": {"end": {"column": 3, "line": 78}, "start": {"column": 3, "line": 78}},
                        "raw": "wbLog(#level, __VA_ARGS__)",
                        "type": "CallExpression"
                    }, {
                        "arguments": [{
                            "cform": "\"TRACE\"",
                            "loc": {"end": {"column": 3, "line": 79}, "start": {"column": 3, "line": 79}},
                            "raw": "TRACE",
                            "type": "StringLiteral",
                            "value": "\"TRACE\""
                        }, {
                            "cform": "\"The grid dimensions are \"",
                            "loc": {"end": {"column": 3, "line": 79}, "start": {"column": 3, "line": 79}},
                            "raw": "The grid dimensions are ",
                            "type": "StringLiteral",
                            "value": "\"The grid dimensions are \""
                        }, {
                            "cform": "gridDim.x",
                            "left": {
                                "cform": "gridDim",
                                "kind": {
                                    "address_spaces": [],
                                    "bases": [{
                                        "cform": "struct dim3",
                                        "loc": {"end": {"column": 54, "line": 76}, "start": {"column": 3, "line": 75}},
                                        "raw": "struct dim3",
                                        "type": "Literal",
                                        "value": "struct dim3"
                                    }],
                                    "cform": "struct dim3 ",
                                    "loc": {"end": {"column": 54, "line": 76}, "start": {"column": 3, "line": 75}},
                                    "qualifiers": [],
                                    "raw": "lockDim.y))...",
                                    "type": "TypeSpecification"
                                },
                                "loc": {"end": {"column": 54, "line": 76}, "start": {"column": 3, "line": 75}},
                                "name": "gridDim",
                                "raw": "lockDim.y))...",
                                "type": "Identifier"
                            },
                            "loc": {"end": {"column": 3, "line": 79}, "start": {"column": 3, "line": 79}},
                            "operator": ".",
                            "raw": "gridDim.x",
                            "right": {
                                "cform": "x",
                                "kind": {
                                    "address_spaces": [],
                                    "bases": [{
                                        "cform": "unsigned int",
                                        "loc": {"end": {"column": 16, "line": 16}, "start": {"column": 3, "line": 16}},
                                        "raw": "unsigned int",
                                        "type": "Literal",
                                        "value": "unsigned int"
                                    }],
                                    "cform": "unsigned int ",
                                    "loc": {"end": {"column": 16, "line": 16}, "start": {"column": 3, "line": 16}},
                                    "qualifiers": [],
                                    "raw": "unsigned int x",
                                    "type": "TypeSpecification"
                                },
                                "loc": {"end": {"column": 16, "line": 16}, "start": {"column": 3, "line": 16}},
                                "name": "x",
                                "raw": "unsigned int x",
                                "type": "Identifier"
                            },
                            "type": "MemberExpression"
                        }, {
                            "cform": "\" x \"",
                            "loc": {"end": {"column": 3, "line": 79}, "start": {"column": 3, "line": 79}},
                            "raw": " x ",
                            "type": "StringLiteral",
                            "value": "\" x \""
                        }, {
                            "cform": "gridDim.y",
                            "left": {
                                "cform": "gridDim",
                                "kind": {
                                    "address_spaces": [],
                                    "bases": [{
                                        "cform": "struct dim3",
                                        "loc": {"end": {"column": 54, "line": 76}, "start": {"column": 3, "line": 75}},
                                        "raw": "struct dim3",
                                        "type": "Literal",
                                        "value": "struct dim3"
                                    }],
                                    "cform": "struct dim3 ",
                                    "loc": {"end": {"column": 54, "line": 76}, "start": {"column": 3, "line": 75}},
                                    "qualifiers": [],
                                    "raw": "lockDim.y))...",
                                    "type": "TypeSpecification"
                                },
                                "loc": {"end": {"column": 54, "line": 76}, "start": {"column": 3, "line": 75}},
                                "name": "gridDim",
                                "raw": "lockDim.y))...",
                                "type": "Identifier"
                            },
                            "loc": {"end": {"column": 3, "line": 79}, "start": {"column": 3, "line": 79}},
                            "operator": ".",
                            "raw": "gridDim.y",
                            "right": {
                                "cform": "y",
                                "kind": {
                                    "address_spaces": [],
                                    "bases": [{
                                        "cform": "unsigned int",
                                        "loc": {"end": {"column": 19, "line": 16}, "start": {"column": 3, "line": 16}},
                                        "raw": "unsigned int",
                                        "type": "Literal",
                                        "value": "unsigned int"
                                    }],
                                    "cform": "unsigned int ",
                                    "loc": {"end": {"column": 19, "line": 16}, "start": {"column": 3, "line": 16}},
                                    "qualifiers": [],
                                    "raw": "unsigned int x, y",
                                    "type": "TypeSpecification"
                                },
                                "loc": {"end": {"column": 19, "line": 16}, "start": {"column": 3, "line": 16}},
                                "name": "y",
                                "raw": "unsigned int x, y",
                                "type": "Identifier"
                            },
                            "type": "MemberExpression"
                        }],
                        "callee": {
                            "cform": "wbLog",
                            "loc": {"end": {"column": 3, "line": 79}, "start": {"column": 3, "line": 79}},
                            "name": "wbLog",
                            "raw": "wbLog(#level, __VA_ARGS__)",
                            "type": "Identifier"
                        },
                        "cform": "wbLog(\"TRACE\" /* String*/, \"The grid dimensions are \" /* String*/, gridDim.x /* Member*/, \" x \" /* String*/, gridDim.y)",
                        "loc": {"end": {"column": 3, "line": 79}, "start": {"column": 3, "line": 79}},
                        "raw": "wbLog(#level, __VA_ARGS__)",
                        "type": "CallExpression"
                    }, {
                        "arguments": [{
                            "cform": "\"Compute\"",
                            "loc": {"end": {"column": 3, "line": 81}, "start": {"column": 3, "line": 81}},
                            "raw": "Compute",
                            "type": "StringLiteral",
                            "value": "\"Compute\""
                        }, {
                            "cform": "\"Performing CUDA computation\"",
                            "loc": {"end": {"column": 3, "line": 81}, "start": {"column": 3, "line": 81}},
                            "raw": "Performing CUDA computation",
                            "type": "StringLiteral",
                            "value": "\"Performing CUDA computation\""
                        }],
                        "callee": {
                            "cform": "wbTime_start",
                            "loc": {"end": {"column": 3, "line": 81}, "start": {"column": 3, "line": 81}},
                            "name": "wbTime_start",
                            "raw": "wbTime_start(#kind, __VA_ARGS__)",
                            "type": "Identifier"
                        },
                        "cform": "wbTime_start(\"Compute\" /* String*/, \"Performing CUDA computation\")",
                        "loc": {"end": {"column": 3, "line": 81}, "start": {"column": 3, "line": 81}},
                        "raw": "wbTime_start(#kind, __VA_ARGS__)",
                        "type": "CallExpression"
                    }, {}, {}, {
                        "arguments": [{
                            "cform": "deviceA",
                            "kind": {
                                "cform": "float *",
                                "loc": {"end": {"column": 10, "line": 27}, "start": {"column": 3, "line": 27}},
                                "raw": "float *deviceA",
                                "type": "ReferenceType",
                                "value": {
                                    "address_spaces": [],
                                    "bases": [{
                                        "cform": "float",
                                        "loc": {"end": {"column": 10, "line": 27}, "start": {"column": 3, "line": 27}},
                                        "raw": "float",
                                        "type": "Literal",
                                        "value": "float"
                                    }],
                                    "cform": "float ",
                                    "loc": {"end": {"column": 10, "line": 27}, "start": {"column": 3, "line": 27}},
                                    "qualifiers": [],
                                    "raw": "float *deviceA",
                                    "type": "TypeSpecification"
                                }
                            },
                            "loc": {"end": {"column": 10, "line": 27}, "start": {"column": 3, "line": 27}},
                            "name": "deviceA",
                            "raw": "float *deviceA",
                            "type": "Identifier"
                        }, {
                            "cform": "deviceB",
                            "kind": {
                                "cform": "float *",
                                "loc": {"end": {"column": 10, "line": 28}, "start": {"column": 3, "line": 28}},
                                "raw": "float *deviceB",
                                "type": "ReferenceType",
                                "value": {
                                    "address_spaces": [],
                                    "bases": [{
                                        "cform": "float",
                                        "loc": {"end": {"column": 10, "line": 28}, "start": {"column": 3, "line": 28}},
                                        "raw": "float",
                                        "type": "Literal",
                                        "value": "float"
                                    }],
                                    "cform": "float ",
                                    "loc": {"end": {"column": 10, "line": 28}, "start": {"column": 3, "line": 28}},
                                    "qualifiers": [],
                                    "raw": "float *deviceB",
                                    "type": "TypeSpecification"
                                }
                            },
                            "loc": {"end": {"column": 10, "line": 28}, "start": {"column": 3, "line": 28}},
                            "name": "deviceB",
                            "raw": "float *deviceB",
                            "type": "Identifier"
                        }, {
                            "cform": "deviceC",
                            "kind": {
                                "cform": "float *",
                                "loc": {"end": {"column": 10, "line": 29}, "start": {"column": 3, "line": 29}},
                                "raw": "float *deviceC",
                                "type": "ReferenceType",
                                "value": {
                                    "address_spaces": [],
                                    "bases": [{
                                        "cform": "float",
                                        "loc": {"end": {"column": 10, "line": 29}, "start": {"column": 3, "line": 29}},
                                        "raw": "float",
                                        "type": "Literal",
                                        "value": "float"
                                    }],
                                    "cform": "float ",
                                    "loc": {"end": {"column": 10, "line": 29}, "start": {"column": 3, "line": 29}},
                                    "qualifiers": [],
                                    "raw": "float *deviceC",
                                    "type": "TypeSpecification"
                                }
                            },
                            "loc": {"end": {"column": 10, "line": 29}, "start": {"column": 3, "line": 29}},
                            "name": "deviceC",
                            "raw": "float *deviceC",
                            "type": "Identifier"
                        }, {
                            "cform": "numARows",
                            "kind": {
                                "address_spaces": [],
                                "bases": [{
                                    "cform": "int",
                                    "loc": {"end": {"column": 7, "line": 30}, "start": {"column": 3, "line": 30}},
                                    "raw": "int",
                                    "type": "Literal",
                                    "value": "int"
                                }],
                                "cform": "int ",
                                "loc": {"end": {"column": 7, "line": 30}, "start": {"column": 3, "line": 30}},
                                "qualifiers": [],
                                "raw": "int numARows",
                                "type": "TypeSpecification"
                            },
                            "loc": {"end": {"column": 7, "line": 30}, "start": {"column": 3, "line": 30}},
                            "name": "numARows",
                            "raw": "int numARows",
                            "type": "Identifier"
                        }, {
                            "cform": "numAColumns",
                            "kind": {
                                "address_spaces": [],
                                "bases": [{
                                    "cform": "int",
                                    "loc": {"end": {"column": 7, "line": 31}, "start": {"column": 3, "line": 31}},
                                    "raw": "int",
                                    "type": "Literal",
                                    "value": "int"
                                }],
                                "cform": "int ",
                                "loc": {"end": {"column": 7, "line": 31}, "start": {"column": 3, "line": 31}},
                                "qualifiers": [],
                                "raw": "int numAColumns",
                                "type": "TypeSpecification"
                            },
                            "loc": {"end": {"column": 7, "line": 31}, "start": {"column": 3, "line": 31}},
                            "name": "numAColumns",
                            "raw": "int numAColumns",
                            "type": "Identifier"
                        }, {
                            "cform": "numBRows",
                            "kind": {
                                "address_spaces": [],
                                "bases": [{
                                    "cform": "int",
                                    "loc": {"end": {"column": 7, "line": 32}, "start": {"column": 3, "line": 32}},
                                    "raw": "int",
                                    "type": "Literal",
                                    "value": "int"
                                }],
                                "cform": "int ",
                                "loc": {"end": {"column": 7, "line": 32}, "start": {"column": 3, "line": 32}},
                                "qualifiers": [],
                                "raw": "int numBRows",
                                "type": "TypeSpecification"
                            },
                            "loc": {"end": {"column": 7, "line": 32}, "start": {"column": 3, "line": 32}},
                            "name": "numBRows",
                            "raw": "int numBRows",
                            "type": "Identifier"
                        }, {
                            "cform": "numBColumns",
                            "kind": {
                                "address_spaces": [],
                                "bases": [{
                                    "cform": "int",
                                    "loc": {"end": {"column": 7, "line": 33}, "start": {"column": 3, "line": 33}},
                                    "raw": "int",
                                    "type": "Literal",
                                    "value": "int"
                                }],
                                "cform": "int ",
                                "loc": {"end": {"column": 7, "line": 33}, "start": {"column": 3, "line": 33}},
                                "qualifiers": [],
                                "raw": "int numBColumns",
                                "type": "TypeSpecification"
                            },
                            "loc": {"end": {"column": 7, "line": 33}, "start": {"column": 3, "line": 33}},
                            "name": "numBColumns",
                            "raw": "int numBColumns",
                            "type": "Identifier"
                        }],
                        "callee": "sgemm",
                        "cform": "sgemm<<<gridDim /* Identifier*/, blockDim>>>(deviceA /* Identifier*/, deviceB /* Identifier*/, deviceC /* Identifier*/, numARows /* Identifier*/, numAColumns /* Identifier*/, numBRows /* Identifier*/, numBColumns)",
                        "config": [{
                            "cform": "gridDim",
                            "kind": {
                                "address_spaces": [],
                                "bases": [{
                                    "cform": "struct dim3",
                                    "loc": {"end": {"column": 54, "line": 76}, "start": {"column": 3, "line": 75}},
                                    "raw": "struct dim3",
                                    "type": "Literal",
                                    "value": "struct dim3"
                                }],
                                "cform": "struct dim3 ",
                                "loc": {"end": {"column": 54, "line": 76}, "start": {"column": 3, "line": 75}},
                                "qualifiers": [],
                                "raw": "lockDim.y))...",
                                "type": "TypeSpecification"
                            },
                            "loc": {"end": {"column": 54, "line": 76}, "start": {"column": 3, "line": 75}},
                            "name": "gridDim",
                            "raw": "lockDim.y))...",
                            "type": "Identifier"
                        }, {
                            "cform": "blockDim",
                            "kind": {
                                "address_spaces": [],
                                "bases": [{
                                    "cform": "struct dim3",
                                    "loc": {"end": {"column": 23, "line": 74}, "start": {"column": 3, "line": 74}},
                                    "raw": "struct dim3",
                                    "type": "Literal",
                                    "value": "struct dim3"
                                }],
                                "cform": "struct dim3 ",
                                "loc": {"end": {"column": 23, "line": 74}, "start": {"column": 3, "line": 74}},
                                "qualifiers": [],
                                "raw": "dim3 blockDim(16, 16)",
                                "type": "TypeSpecification"
                            },
                            "loc": {"end": {"column": 23, "line": 74}, "start": {"column": 3, "line": 74}},
                            "name": "blockDim",
                            "raw": "dim3 blockDim(16, 16)",
                            "type": "Identifier"
                        }],
                        "loc": {"end": {"column": 79, "line": 85}, "start": {"column": 3, "line": 84}},
                        "raw": "numBColumns)...",
                        "type": "CallExpression"
                    }, {
                        "arguments": [],
                        "callee": {
                            "cform": "cudaDeviceSynchronize",
                            "loc": {"end": {"column": 25, "line": 86}, "start": {"column": 3, "line": 86}},
                            "name": "cudaDeviceSynchronize",
                            "raw": "cudaDeviceSynchronize()",
                            "type": "Identifier"
                        },
                        "cform": "cudaDeviceSynchronize()",
                        "loc": {"end": {"column": 25, "line": 86}, "start": {"column": 3, "line": 86}},
                        "raw": "cudaDeviceSynchronize()",
                        "type": "CallExpression"
                    }, {
                        "arguments": [{
                            "cform": "\"Compute\"",
                            "loc": {"end": {"column": 3, "line": 87}, "start": {"column": 3, "line": 87}},
                            "raw": "Compute",
                            "type": "StringLiteral",
                            "value": "\"Compute\""
                        }, {
                            "cform": "\"Performing CUDA computation\"",
                            "loc": {"end": {"column": 3, "line": 87}, "start": {"column": 3, "line": 87}},
                            "raw": "Performing CUDA computation",
                            "type": "StringLiteral",
                            "value": "\"Performing CUDA computation\""
                        }],
                        "callee": {
                            "cform": "wbTime_stop",
                            "loc": {"end": {"column": 3, "line": 87}, "start": {"column": 3, "line": 87}},
                            "name": "wbTime_stop",
                            "raw": "wbTime_stop(#kind, __VA_ARGS__)",
                            "type": "Identifier"
                        },
                        "cform": "wbTime_stop(\"Compute\" /* String*/, \"Performing CUDA computation\")",
                        "loc": {"end": {"column": 3, "line": 87}, "start": {"column": 3, "line": 87}},
                        "raw": "wbTime_stop(#kind, __VA_ARGS__)",
                        "type": "CallExpression"
                    }, {}, {
                        "arguments": [{
                            "cform": "\"Copy\"",
                            "loc": {"end": {"column": 3, "line": 89}, "start": {"column": 3, "line": 89}},
                            "raw": "Copy",
                            "type": "StringLiteral",
                            "value": "\"Copy\""
                        }, {
                            "cform": "\"Copying output memory to the CPU\"",
                            "loc": {"end": {"column": 3, "line": 89}, "start": {"column": 3, "line": 89}},
                            "raw": "Copying output memory to the CPU",
                            "type": "StringLiteral",
                            "value": "\"Copying output memory to the CPU\""
                        }],
                        "callee": {
                            "cform": "wbTime_start",
                            "loc": {"end": {"column": 3, "line": 89}, "start": {"column": 3, "line": 89}},
                            "name": "wbTime_start",
                            "raw": "wbTime_start(#kind, __VA_ARGS__)",
                            "type": "Identifier"
                        },
                        "cform": "wbTime_start(\"Copy\" /* String*/, \"Copying output memory to the CPU\")",
                        "loc": {"end": {"column": 3, "line": 89}, "start": {"column": 3, "line": 89}},
                        "raw": "wbTime_start(#kind, __VA_ARGS__)",
                        "type": "CallExpression"
                    }, {}, {}, {
                        "arguments": [{
                            "cform": "\"Copy\"",
                            "loc": {"end": {"column": 3, "line": 94}, "start": {"column": 3, "line": 94}},
                            "raw": "Copy",
                            "type": "StringLiteral",
                            "value": "\"Copy\""
                        }, {
                            "cform": "\"Copying output memory to the CPU\"",
                            "loc": {"end": {"column": 3, "line": 94}, "start": {"column": 3, "line": 94}},
                            "raw": "Copying output memory to the CPU",
                            "type": "StringLiteral",
                            "value": "\"Copying output memory to the CPU\""
                        }],
                        "callee": {
                            "cform": "wbTime_stop",
                            "loc": {"end": {"column": 3, "line": 94}, "start": {"column": 3, "line": 94}},
                            "name": "wbTime_stop",
                            "raw": "wbTime_stop(#kind, __VA_ARGS__)",
                            "type": "Identifier"
                        },
                        "cform": "wbTime_stop(\"Copy\" /* String*/, \"Copying output memory to the CPU\")",
                        "loc": {"end": {"column": 3, "line": 94}, "start": {"column": 3, "line": 94}},
                        "raw": "wbTime_stop(#kind, __VA_ARGS__)",
                        "type": "CallExpression"
                    }, {}, {
                        "arguments": [{
                            "cform": "\"GPU\"",
                            "loc": {"end": {"column": 3, "line": 96}, "start": {"column": 3, "line": 96}},
                            "raw": "GPU",
                            "type": "StringLiteral",
                            "value": "\"GPU\""
                        }, {
                            "cform": "\"Freeing GPU Memory\"",
                            "loc": {"end": {"column": 3, "line": 96}, "start": {"column": 3, "line": 96}},
                            "raw": "Freeing GPU Memory",
                            "type": "StringLiteral",
                            "value": "\"Freeing GPU Memory\""
                        }],
                        "callee": {
                            "cform": "wbTime_start",
                            "loc": {"end": {"column": 3, "line": 96}, "start": {"column": 3, "line": 96}},
                            "name": "wbTime_start",
                            "raw": "wbTime_start(#kind, __VA_ARGS__)",
                            "type": "Identifier"
                        },
                        "cform": "wbTime_start(\"GPU\" /* String*/, \"Freeing GPU Memory\")",
                        "loc": {"end": {"column": 3, "line": 96}, "start": {"column": 3, "line": 96}},
                        "raw": "wbTime_start(#kind, __VA_ARGS__)",
                        "type": "CallExpression"
                    }, {}, {
                        "arguments": [{
                            "cform": "deviceA",
                            "kind": {
                                "cform": "float *",
                                "loc": {"end": {"column": 10, "line": 27}, "start": {"column": 3, "line": 27}},
                                "raw": "float *deviceA",
                                "type": "ReferenceType",
                                "value": {
                                    "address_spaces": [],
                                    "bases": [{
                                        "cform": "float",
                                        "loc": {"end": {"column": 10, "line": 27}, "start": {"column": 3, "line": 27}},
                                        "raw": "float",
                                        "type": "Literal",
                                        "value": "float"
                                    }],
                                    "cform": "float ",
                                    "loc": {"end": {"column": 10, "line": 27}, "start": {"column": 3, "line": 27}},
                                    "qualifiers": [],
                                    "raw": "float *deviceA",
                                    "type": "TypeSpecification"
                                }
                            },
                            "loc": {"end": {"column": 10, "line": 27}, "start": {"column": 3, "line": 27}},
                            "name": "deviceA",
                            "raw": "float *deviceA",
                            "type": "Identifier"
                        }],
                        "callee": {
                            "cform": "cudaFree",
                            "loc": {"end": {"column": 19, "line": 98}, "start": {"column": 3, "line": 98}},
                            "name": "cudaFree",
                            "raw": "cudaFree(deviceA)",
                            "type": "Identifier"
                        },
                        "cform": "cudaFree(deviceA)",
                        "loc": {"end": {"column": 19, "line": 98}, "start": {"column": 3, "line": 98}},
                        "raw": "cudaFree(deviceA)",
                        "type": "CallExpression"
                    }, {
                        "arguments": [{
                            "cform": "deviceB",
                            "kind": {
                                "cform": "float *",
                                "loc": {"end": {"column": 10, "line": 28}, "start": {"column": 3, "line": 28}},
                                "raw": "float *deviceB",
                                "type": "ReferenceType",
                                "value": {
                                    "address_spaces": [],
                                    "bases": [{
                                        "cform": "float",
                                        "loc": {"end": {"column": 10, "line": 28}, "start": {"column": 3, "line": 28}},
                                        "raw": "float",
                                        "type": "Literal",
                                        "value": "float"
                                    }],
                                    "cform": "float ",
                                    "loc": {"end": {"column": 10, "line": 28}, "start": {"column": 3, "line": 28}},
                                    "qualifiers": [],
                                    "raw": "float *deviceB",
                                    "type": "TypeSpecification"
                                }
                            },
                            "loc": {"end": {"column": 10, "line": 28}, "start": {"column": 3, "line": 28}},
                            "name": "deviceB",
                            "raw": "float *deviceB",
                            "type": "Identifier"
                        }],
                        "callee": {
                            "cform": "cudaFree",
                            "loc": {"end": {"column": 19, "line": 99}, "start": {"column": 3, "line": 99}},
                            "name": "cudaFree",
                            "raw": "cudaFree(deviceB)",
                            "type": "Identifier"
                        },
                        "cform": "cudaFree(deviceB)",
                        "loc": {"end": {"column": 19, "line": 99}, "start": {"column": 3, "line": 99}},
                        "raw": "cudaFree(deviceB)",
                        "type": "CallExpression"
                    }, {
                        "arguments": [{
                            "cform": "deviceC",
                            "kind": {
                                "cform": "float *",
                                "loc": {"end": {"column": 10, "line": 29}, "start": {"column": 3, "line": 29}},
                                "raw": "float *deviceC",
                                "type": "ReferenceType",
                                "value": {
                                    "address_spaces": [],
                                    "bases": [{
                                        "cform": "float",
                                        "loc": {"end": {"column": 10, "line": 29}, "start": {"column": 3, "line": 29}},
                                        "raw": "float",
                                        "type": "Literal",
                                        "value": "float"
                                    }],
                                    "cform": "float ",
                                    "loc": {"end": {"column": 10, "line": 29}, "start": {"column": 3, "line": 29}},
                                    "qualifiers": [],
                                    "raw": "float *deviceC",
                                    "type": "TypeSpecification"
                                }
                            },
                            "loc": {"end": {"column": 10, "line": 29}, "start": {"column": 3, "line": 29}},
                            "name": "deviceC",
                            "raw": "float *deviceC",
                            "type": "Identifier"
                        }],
                        "callee": {
                            "cform": "cudaFree",
                            "loc": {"end": {"column": 19, "line": 100}, "start": {"column": 3, "line": 100}},
                            "name": "cudaFree",
                            "raw": "cudaFree(deviceC)",
                            "type": "Identifier"
                        },
                        "cform": "cudaFree(deviceC)",
                        "loc": {"end": {"column": 19, "line": 100}, "start": {"column": 3, "line": 100}},
                        "raw": "cudaFree(deviceC)",
                        "type": "CallExpression"
                    }, {
                        "arguments": [{
                            "cform": "\"GPU\"",
                            "loc": {"end": {"column": 3, "line": 101}, "start": {"column": 3, "line": 101}},
                            "raw": "GPU",
                            "type": "StringLiteral",
                            "value": "\"GPU\""
                        }, {
                            "cform": "\"Freeing GPU Memory\"",
                            "loc": {"end": {"column": 3, "line": 101}, "start": {"column": 3, "line": 101}},
                            "raw": "Freeing GPU Memory",
                            "type": "StringLiteral",
                            "value": "\"Freeing GPU Memory\""
                        }],
                        "callee": {
                            "cform": "wbTime_stop",
                            "loc": {"end": {"column": 3, "line": 101}, "start": {"column": 3, "line": 101}},
                            "name": "wbTime_stop",
                            "raw": "wbTime_stop(#kind, __VA_ARGS__)",
                            "type": "Identifier"
                        },
                        "cform": "wbTime_stop(\"GPU\" /* String*/, \"Freeing GPU Memory\")",
                        "loc": {"end": {"column": 3, "line": 101}, "start": {"column": 3, "line": 101}},
                        "raw": "wbTime_stop(#kind, __VA_ARGS__)",
                        "type": "CallExpression"
                    }, {}, {
                        "arguments": [{
                            "cform": "args",
                            "kind": {
                                "address_spaces": [],
                                "bases": [{
                                    "cform": "int",
                                    "loc": {"end": {"column": 11, "line": 23}, "start": {"column": 3, "line": 23}},
                                    "raw": "int",
                                    "type": "Literal",
                                    "value": "int"
                                }],
                                "cform": "int ",
                                "loc": {"end": {"column": 11, "line": 23}, "start": {"column": 3, "line": 23}},
                                "qualifiers": [],
                                "raw": "wbArg_t args",
                                "type": "TypeSpecification"
                            },
                            "loc": {"end": {"column": 11, "line": 23}, "start": {"column": 3, "line": 23}},
                            "name": "args",
                            "raw": "wbArg_t args",
                            "type": "Identifier"
                        }, {
                            "cform": "hostC",
                            "kind": {
                                "cform": "float *",
                                "loc": {"end": {"column": 10, "line": 26}, "start": {"column": 3, "line": 26}},
                                "raw": "float *hostC",
                                "type": "ReferenceType",
                                "value": {
                                    "address_spaces": [],
                                    "bases": [{
                                        "cform": "float",
                                        "loc": {"end": {"column": 10, "line": 26}, "start": {"column": 3, "line": 26}},
                                        "raw": "float",
                                        "type": "Literal",
                                        "value": "float"
                                    }],
                                    "cform": "float ",
                                    "loc": {"end": {"column": 10, "line": 26}, "start": {"column": 3, "line": 26}},
                                    "qualifiers": [],
                                    "raw": "float *hostC",
                                    "type": "TypeSpecification"
                                }
                            },
                            "loc": {"end": {"column": 10, "line": 26}, "start": {"column": 3, "line": 26}},
                            "name": "hostC",
                            "raw": "float *hostC",
                            "type": "Identifier"
                        }, {
                            "cform": "numARows",
                            "kind": {
                                "address_spaces": [],
                                "bases": [{
                                    "cform": "int",
                                    "loc": {"end": {"column": 7, "line": 30}, "start": {"column": 3, "line": 30}},
                                    "raw": "int",
                                    "type": "Literal",
                                    "value": "int"
                                }],
                                "cform": "int ",
                                "loc": {"end": {"column": 7, "line": 30}, "start": {"column": 3, "line": 30}},
                                "qualifiers": [],
                                "raw": "int numARows",
                                "type": "TypeSpecification"
                            },
                            "loc": {"end": {"column": 7, "line": 30}, "start": {"column": 3, "line": 30}},
                            "name": "numARows",
                            "raw": "int numARows",
                            "type": "Identifier"
                        }, {
                            "cform": "numBColumns",
                            "kind": {
                                "address_spaces": [],
                                "bases": [{
                                    "cform": "int",
                                    "loc": {"end": {"column": 7, "line": 33}, "start": {"column": 3, "line": 33}},
                                    "raw": "int",
                                    "type": "Literal",
                                    "value": "int"
                                }],
                                "cform": "int ",
                                "loc": {"end": {"column": 7, "line": 33}, "start": {"column": 3, "line": 33}},
                                "qualifiers": [],
                                "raw": "int numBColumns",
                                "type": "TypeSpecification"
                            },
                            "loc": {"end": {"column": 7, "line": 33}, "start": {"column": 3, "line": 33}},
                            "name": "numBColumns",
                            "raw": "int numBColumns",
                            "type": "Identifier"
                        }],
                        "callee": {
                            "cform": "wbSolution",
                            "loc": {"end": {"column": 48, "line": 103}, "start": {"column": 3, "line": 103}},
                            "name": "wbSolution",
                            "raw": "wbSolution(args, hostC, numARows, numBColumns)",
                            "type": "Identifier"
                        },
                        "cform": "wbSolution(args /* Identifier*/, hostC /* Identifier*/, numARows /* Identifier*/, numBColumns)",
                        "loc": {"end": {"column": 48, "line": 103}, "start": {"column": 3, "line": 103}},
                        "raw": "wbSolution(args, hostC, numARows, numBColumns)",
                        "type": "CallExpression"
                    }, {
                        "arguments": [{
                            "cform": "hostA",
                            "kind": {
                                "cform": "float *",
                                "loc": {"end": {"column": 10, "line": 24}, "start": {"column": 3, "line": 24}},
                                "raw": "float *hostA",
                                "type": "ReferenceType",
                                "value": {
                                    "address_spaces": [],
                                    "bases": [{
                                        "cform": "float",
                                        "loc": {"end": {"column": 10, "line": 24}, "start": {"column": 3, "line": 24}},
                                        "raw": "float",
                                        "type": "Literal",
                                        "value": "float"
                                    }],
                                    "cform": "float ",
                                    "loc": {"end": {"column": 10, "line": 24}, "start": {"column": 3, "line": 24}},
                                    "qualifiers": [],
                                    "raw": "float *hostA",
                                    "type": "TypeSpecification"
                                }
                            },
                            "loc": {"end": {"column": 10, "line": 24}, "start": {"column": 3, "line": 24}},
                            "name": "hostA",
                            "raw": "float *hostA",
                            "type": "Identifier"
                        }],
                        "callee": {
                            "cform": "free",
                            "loc": {"end": {"column": 13, "line": 105}, "start": {"column": 3, "line": 105}},
                            "name": "free",
                            "raw": "free(hostA)",
                            "type": "Identifier"
                        },
                        "cform": "free(hostA)",
                        "loc": {"end": {"column": 13, "line": 105}, "start": {"column": 3, "line": 105}},
                        "raw": "free(hostA)",
                        "type": "CallExpression"
                    }, {
                        "arguments": [{
                            "cform": "hostB",
                            "kind": {
                                "cform": "float *",
                                "loc": {"end": {"column": 10, "line": 25}, "start": {"column": 3, "line": 25}},
                                "raw": "float *hostB",
                                "type": "ReferenceType",
                                "value": {
                                    "address_spaces": [],
                                    "bases": [{
                                        "cform": "float",
                                        "loc": {"end": {"column": 10, "line": 25}, "start": {"column": 3, "line": 25}},
                                        "raw": "float",
                                        "type": "Literal",
                                        "value": "float"
                                    }],
                                    "cform": "float ",
                                    "loc": {"end": {"column": 10, "line": 25}, "start": {"column": 3, "line": 25}},
                                    "qualifiers": [],
                                    "raw": "float *hostB",
                                    "type": "TypeSpecification"
                                }
                            },
                            "loc": {"end": {"column": 10, "line": 25}, "start": {"column": 3, "line": 25}},
                            "name": "hostB",
                            "raw": "float *hostB",
                            "type": "Identifier"
                        }],
                        "callee": {
                            "cform": "free",
                            "loc": {"end": {"column": 13, "line": 106}, "start": {"column": 3, "line": 106}},
                            "name": "free",
                            "raw": "free(hostB)",
                            "type": "Identifier"
                        },
                        "cform": "free(hostB)",
                        "loc": {"end": {"column": 13, "line": 106}, "start": {"column": 3, "line": 106}},
                        "raw": "free(hostB)",
                        "type": "CallExpression"
                    }, {
                        "arguments": [{
                            "cform": "hostC",
                            "kind": {
                                "cform": "float *",
                                "loc": {"end": {"column": 10, "line": 26}, "start": {"column": 3, "line": 26}},
                                "raw": "float *hostC",
                                "type": "ReferenceType",
                                "value": {
                                    "address_spaces": [],
                                    "bases": [{
                                        "cform": "float",
                                        "loc": {"end": {"column": 10, "line": 26}, "start": {"column": 3, "line": 26}},
                                        "raw": "float",
                                        "type": "Literal",
                                        "value": "float"
                                    }],
                                    "cform": "float ",
                                    "loc": {"end": {"column": 10, "line": 26}, "start": {"column": 3, "line": 26}},
                                    "qualifiers": [],
                                    "raw": "float *hostC",
                                    "type": "TypeSpecification"
                                }
                            },
                            "loc": {"end": {"column": 10, "line": 26}, "start": {"column": 3, "line": 26}},
                            "name": "hostC",
                            "raw": "float *hostC",
                            "type": "Identifier"
                        }],
                        "callee": {
                            "cform": "free",
                            "loc": {"end": {"column": 13, "line": 107}, "start": {"column": 3, "line": 107}},
                            "name": "free",
                            "raw": "free(hostC)",
                            "type": "Identifier"
                        },
                        "cform": "free(hostC)",
                        "loc": {"end": {"column": 13, "line": 107}, "start": {"column": 3, "line": 107}},
                        "raw": "free(hostC)",
                        "type": "CallExpression"
                    }, {
                        "argument": {
                            "cform": "0",
                            "loc": {"end": {"column": 10, "line": 109}, "start": {"column": 10, "line": 109}},
                            "raw": "0",
                            "type": "Integer32Literal",
                            "value": 0
                        },
                        "cform": "return 0",
                        "loc": {"end": {"column": 10, "line": 109}, "start": {"column": 3, "line": 109}},
                        "raw": "return 0",
                        "type": "ReturnStatement"
                    }],
                    "cform": "{\nint  args; /* Declare*/\nfloat * hostA; /* Declare*/\nfloat * hostB; /* Declare*/\nfloat * hostC; /* Declare*/\nfloat * deviceA; /* Declare*/\nfloat * deviceB; /* Declare*/\nfloat * deviceC; /* Declare*/\nint  numARows; /* Declare*/\nint  numAColumns; /* Declare*/\nint  numBRows; /* Declare*/\nint  numBColumns; /* Declare*/\nint  numCRows; /* Declare*/\nint  numCColumns; /* Declare*/\nargs = wbArg_read(argc /* Identifier*/, argv); /* Assign*/\nwbTime_start(\"Generic\" /* String*/, \"Importing data and creating memory on host\"); /* Call*/\nhostA = wbImport(\"input0\" /* String*/, & numARows /* UnaryOperator*/, & numAColumns); /* Assign*/\nhostB = wbImport(\"input1\" /* String*/, & numBRows /* UnaryOperator*/, & numBColumns); /* Assign*/\nhostC = malloc(numARows * numBColumns * sizeof(float )); /* Assign*/\nwbTime_stop(\"Generic\" /* String*/, \"Importing data and creating memory on host\"); /* Call*/\nnumCRows = numARows; /* Assign*/\nnumCColumns = numBColumns; /* Assign*/\nwbLog(\"TRACE\" /* String*/, \"The dimensions of A are \" /* String*/, numARows /* Identifier*/, \" x \" /* String*/, numAColumns); /* Call*/\nwbLog(\"TRACE\" /* String*/, \"The dimensions of B are \" /* String*/, numBRows /* Identifier*/, \" x \" /* String*/, numBColumns); /* Call*/\nwbLog(\"TRACE\" /* String*/, \"The dimensions of C are \" /* String*/, numCRows /* Identifier*/, \" x \" /* String*/, numCColumns); /* Call*/\nwbTime_start(\"GPU\" /* String*/, \"Allocating GPU memory.\"); /* Call*/\nwbTime_stop(\"GPU\" /* String*/, \"Allocating GPU memory.\"); /* Call*/\nwbTime_start(\"GPU\" /* String*/, \"Copying input memory to the GPU.\"); /* Call*/\nwbTime_stop(\"GPU\" /* String*/, \"Copying input memory to the GPU.\"); /* Call*/\nstruct dim3  blockDim = {16 /* Integer32*/, 16}; /* Declare*/\nstruct dim3  gridDim = {ceil((numAColumns) / blockDim.x) /* Call*/, ceil((numBRows) / blockDim.y)}; /* Declare*/\nwbLog(\"TRACE\" /* String*/, \"The block dimensions are \" /* String*/, blockDim.x /* Member*/, \" x \" /* String*/, blockDim.y); /* Call*/\nwbLog(\"TRACE\" /* String*/, \"The grid dimensions are \" /* String*/, gridDim.x /* Member*/, \" x \" /* String*/, gridDim.y); /* Call*/\nwbTime_start(\"Compute\" /* String*/, \"Performing CUDA computation\"); /* Call*/\nsgemm<<<gridDim /* Identifier*/, blockDim>>>(deviceA /* Identifier*/, deviceB /* Identifier*/, deviceC /* Identifier*/, numARows /* Identifier*/, numAColumns /* Identifier*/, numBRows /* Identifier*/, numBColumns); /* Call*/\ncudaDeviceSynchronize(); /* Call*/\nwbTime_stop(\"Compute\" /* String*/, \"Performing CUDA computation\"); /* Call*/\nwbTime_start(\"Copy\" /* String*/, \"Copying output memory to the CPU\"); /* Call*/\nwbTime_stop(\"Copy\" /* String*/, \"Copying output memory to the CPU\"); /* Call*/\nwbTime_start(\"GPU\" /* String*/, \"Freeing GPU Memory\"); /* Call*/\ncudaFree(deviceA); /* Call*/\ncudaFree(deviceB); /* Call*/\ncudaFree(deviceC); /* Call*/\nwbTime_stop(\"GPU\" /* String*/, \"Freeing GPU Memory\"); /* Call*/\nwbSolution(args /* Identifier*/, hostC /* Identifier*/, numARows /* Identifier*/, numBColumns); /* Call*/\nfree(hostA); /* Call*/\nfree(hostB); /* Call*/\nfree(hostC); /* Call*/\nreturn 0; /* Return*/\n}\n",
                    "loc": {"end": {"column": 1, "line": 110}, "start": {"column": 1, "line": 22}},
                    "raw": "/ The B matrix\n  float *hostC; // The output C matrix\n  float *deviceA;\n  float *deviceB;\n  float *deviceC;\n  int numARows;    // number of rows in the matrix A\n  int numAColumns; // number of columns in the matrix A\n  int numBRows;    // number of rows in the matrix B\n  int numBColumns; // number of columns in the matrix B\n  int numCRows;\n  int numCColumns;\n\n  args = wbArg_read(argc, argv);\n\n  wbTime_start(Generic, \"Importing data and creating memory on host\");\n  hostA =\n      ( float * )wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);\n  hostB =\n      ( float * )wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);\n  //@@ Allocate the hostC matrix\n  hostC = ( float * )malloc(numARows * numBColumns * sizeof(float));\n  wbTime_stop(Generic, \"Importing data and creating memory on host\");\n\n  numCRows = numARows;\n  numCColumns = numBColumns;\n\n  wbLog(TRACE, \"The dimensions of A are \", numARows, \" x \", numAColumns);\n  wbLog(TRACE, \"The dimensions of B are \", numBRows, \" x \", numBColumns);\n  wbLog(TRACE, \"The dimensions of C are \", numCRows, \" x \", numCColumns);\n\n  wbTime_start(GPU, \"Allocating GPU memory.\");\n  //@@ Allocate GPU memory here\n  wbCheck(\n      cudaMalloc(( void ** )&deviceA, numARows * numAColumns * sizeof(float)));\n  wbCheck(\n      cudaMalloc(( void ** )&deviceB, numBRows * numBColumns * sizeof(float)));\n  wbCheck(\n      cudaMalloc(( void ** )&deviceC, numARows * numBColumns * sizeof(float)));\n  wbTime_stop(GPU, \"Allocating GPU memory.\");\n\n  wbTime_start(GPU, \"Copying input memory to the GPU.\");\n  //@@ Copy memory to the GPU here\n  wbCheck(cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float),\n                     cudaMemcpyHostToDevice));\n  wbCheck(cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float),\n                     cudaMemcpyHostToDevice));\n  wbTime_stop(GPU, \"Copying input memory to the GPU.\");\n\n  //@@ Initialize the grid and block dimensions here\n  dim3 blockDim(16, 16);\n  dim3 gridDim(ceil((( float )numAColumns) / blockDim.x),\n               ceil((( float )numBRows) / blockDim.y));\n\n  wbLog(TRACE, \"The block dimensions are \", blockDim.x, \" x \", blockDim.y);\n  wbLog(TRACE, \"The grid dimensions are \", gridDim.x, \" x \", gridDim.y);\n\n  wbTime_start(Compute, \"Performing CUDA computation\");\n  //@@ Launch the GPU Kernel here\n  wbCheck(cudaMemset(deviceC, 0, numARows * numBColumns * sizeof(float)));\n  sgemm <<< gridDim, blockDim >>>\n      (deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);\n  cudaDeviceSynchronize();\n  wbTime_stop(Compute, \"Performing CUDA computation\");\n\n  wbTime_start(Copy, \"Copying output memory to the CPU\");\n  //@@ Copy the GPU memory back to the CPU here\n\n  wbCheck(cudaMemcpy(hostC, deviceC, numARows * numBColumns * sizeof(float),\n                     cudaMemcpyDeviceToHost));\n  wbTime_stop(Copy, \"Copying output memory to the CPU\");\n\n  wbTime_start(GPU, \"Freeing GPU Memory\");\n  //@@ Free the GPU memory here\n  cudaFree(deviceA);\n  cudaFree(deviceB);\n  cudaFree(deviceC);\n  wbTime_stop(GPU, \"Freeing GPU Memory\");\n\n  wbSolution(args, hostC, numARows, numBColumns);\n\n  free(hostA);\n  free(hostB);\n  free(hostC);\n\n  return 0;\n}...",
                    "type": "BlockStatement"
                },
                "cform": "int  main(int  argc /* Parameter*/, char ** argv){\nint  args; /* Declare*/\nfloat * hostA; /* Declare*/\nfloat * hostB; /* Declare*/\nfloat * hostC; /* Declare*/\nfloat * deviceA; /* Declare*/\nfloat * deviceB; /* Declare*/\nfloat * deviceC; /* Declare*/\nint  numARows; /* Declare*/\nint  numAColumns; /* Declare*/\nint  numBRows; /* Declare*/\nint  numBColumns; /* Declare*/\nint  numCRows; /* Declare*/\nint  numCColumns; /* Declare*/\nargs = wbArg_read(argc /* Identifier*/, argv); /* Assign*/\nwbTime_start(\"Generic\" /* String*/, \"Importing data and creating memory on host\"); /* Call*/\nhostA = wbImport(\"input0\" /* String*/, & numARows /* UnaryOperator*/, & numAColumns); /* Assign*/\nhostB = wbImport(\"input1\" /* String*/, & numBRows /* UnaryOperator*/, & numBColumns); /* Assign*/\nhostC = malloc(numARows * numBColumns * sizeof(float )); /* Assign*/\nwbTime_stop(\"Generic\" /* String*/, \"Importing data and creating memory on host\"); /* Call*/\nnumCRows = numARows; /* Assign*/\nnumCColumns = numBColumns; /* Assign*/\nwbLog(\"TRACE\" /* String*/, \"The dimensions of A are \" /* String*/, numARows /* Identifier*/, \" x \" /* String*/, numAColumns); /* Call*/\nwbLog(\"TRACE\" /* String*/, \"The dimensions of B are \" /* String*/, numBRows /* Identifier*/, \" x \" /* String*/, numBColumns); /* Call*/\nwbLog(\"TRACE\" /* String*/, \"The dimensions of C are \" /* String*/, numCRows /* Identifier*/, \" x \" /* String*/, numCColumns); /* Call*/\nwbTime_start(\"GPU\" /* String*/, \"Allocating GPU memory.\"); /* Call*/\nwbTime_stop(\"GPU\" /* String*/, \"Allocating GPU memory.\"); /* Call*/\nwbTime_start(\"GPU\" /* String*/, \"Copying input memory to the GPU.\"); /* Call*/\nwbTime_stop(\"GPU\" /* String*/, \"Copying input memory to the GPU.\"); /* Call*/\nstruct dim3  blockDim = {16 /* Integer32*/, 16}; /* Declare*/\nstruct dim3  gridDim = {ceil((numAColumns) / blockDim.x) /* Call*/, ceil((numBRows) / blockDim.y)}; /* Declare*/\nwbLog(\"TRACE\" /* String*/, \"The block dimensions are \" /* String*/, blockDim.x /* Member*/, \" x \" /* String*/, blockDim.y); /* Call*/\nwbLog(\"TRACE\" /* String*/, \"The grid dimensions are \" /* String*/, gridDim.x /* Member*/, \" x \" /* String*/, gridDim.y); /* Call*/\nwbTime_start(\"Compute\" /* String*/, \"Performing CUDA computation\"); /* Call*/\nsgemm<<<gridDim /* Identifier*/, blockDim>>>(deviceA /* Identifier*/, deviceB /* Identifier*/, deviceC /* Identifier*/, numARows /* Identifier*/, numAColumns /* Identifier*/, numBRows /* Identifier*/, numBColumns); /* Call*/\ncudaDeviceSynchronize(); /* Call*/\nwbTime_stop(\"Compute\" /* String*/, \"Performing CUDA computation\"); /* Call*/\nwbTime_start(\"Copy\" /* String*/, \"Copying output memory to the CPU\"); /* Call*/\nwbTime_stop(\"Copy\" /* String*/, \"Copying output memory to the CPU\"); /* Call*/\nwbTime_start(\"GPU\" /* String*/, \"Freeing GPU Memory\"); /* Call*/\ncudaFree(deviceA); /* Call*/\ncudaFree(deviceB); /* Call*/\ncudaFree(deviceC); /* Call*/\nwbTime_stop(\"GPU\" /* String*/, \"Freeing GPU Memory\"); /* Call*/\nwbSolution(args /* Identifier*/, hostC /* Identifier*/, numARows /* Identifier*/, numBColumns); /* Call*/\nfree(hostA); /* Call*/\nfree(hostB); /* Call*/\nfree(hostC); /* Call*/\nreturn 0; /* Return*/\n}\n",
                "id": "main",
                "loc": {"end": {"column": 1, "line": 110}, "start": {"column": 1, "line": 22}},
                "params": [{
                    "data": {
                        "cform": "argc",
                        "loc": {"end": {"column": 14, "line": 22}, "start": {"column": 10, "line": 22}},
                        "name": "argc",
                        "raw": "int argc",
                        "type": "Identifier"
                    }, "type": "ParameterExpression"
                }, {
                    "data": {
                        "cform": "argv",
                        "loc": {"end": {"column": 27, "line": 22}, "start": {"column": 20, "line": 22}},
                        "name": "argv",
                        "raw": "char **argv",
                        "type": "Identifier"
                    }, "type": "ParameterExpression"
                }],
                "raw": "/ The B matrix\n  float *hostC; // The output C matrix\n  float *deviceA;\n  float *deviceB;\n  float *deviceC;\n  int numARows;    // number of rows in the matrix A\n  int numAColumns; // number of columns in the matrix A\n  int numBRows;    // number of rows in the matrix B\n  int numBColumns; // number of columns in the matrix B\n  int numCRows;\n  int numCColumns;\n\n  args = wbArg_read(argc, argv);\n\n  wbTime_start(Generic, \"Importing data and creating memory on host\");\n  hostA =\n      ( float * )wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);\n  hostB =\n      ( float * )wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);\n  //@@ Allocate the hostC matrix\n  hostC = ( float * )malloc(numARows * numBColumns * sizeof(float));\n  wbTime_stop(Generic, \"Importing data and creating memory on host\");\n\n  numCRows = numARows;\n  numCColumns = numBColumns;\n\n  wbLog(TRACE, \"The dimensions of A are \", numARows, \" x \", numAColumns);\n  wbLog(TRACE, \"The dimensions of B are \", numBRows, \" x \", numBColumns);\n  wbLog(TRACE, \"The dimensions of C are \", numCRows, \" x \", numCColumns);\n\n  wbTime_start(GPU, \"Allocating GPU memory.\");\n  //@@ Allocate GPU memory here\n  wbCheck(\n      cudaMalloc(( void ** )&deviceA, numARows * numAColumns * sizeof(float)));\n  wbCheck(\n      cudaMalloc(( void ** )&deviceB, numBRows * numBColumns * sizeof(float)));\n  wbCheck(\n      cudaMalloc(( void ** )&deviceC, numARows * numBColumns * sizeof(float)));\n  wbTime_stop(GPU, \"Allocating GPU memory.\");\n\n  wbTime_start(GPU, \"Copying input memory to the GPU.\");\n  //@@ Copy memory to the GPU here\n  wbCheck(cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float),\n                     cudaMemcpyHostToDevice));\n  wbCheck(cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float),\n                     cudaMemcpyHostToDevice));\n  wbTime_stop(GPU, \"Copying input memory to the GPU.\");\n\n  //@@ Initialize the grid and block dimensions here\n  dim3 blockDim(16, 16);\n  dim3 gridDim(ceil((( float )numAColumns) / blockDim.x),\n               ceil((( float )numBRows) / blockDim.y));\n\n  wbLog(TRACE, \"The block dimensions are \", blockDim.x, \" x \", blockDim.y);\n  wbLog(TRACE, \"The grid dimensions are \", gridDim.x, \" x \", gridDim.y);\n\n  wbTime_start(Compute, \"Performing CUDA computation\");\n  //@@ Launch the GPU Kernel here\n  wbCheck(cudaMemset(deviceC, 0, numARows * numBColumns * sizeof(float)));\n  sgemm <<< gridDim, blockDim >>>\n      (deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);\n  cudaDeviceSynchronize();\n  wbTime_stop(Compute, \"Performing CUDA computation\");\n\n  wbTime_start(Copy, \"Copying output memory to the CPU\");\n  //@@ Copy the GPU memory back to the CPU here\n\n  wbCheck(cudaMemcpy(hostC, deviceC, numARows * numBColumns * sizeof(float),\n                     cudaMemcpyDeviceToHost));\n  wbTime_stop(Copy, \"Copying output memory to the CPU\");\n\n  wbTime_start(GPU, \"Freeing GPU Memory\");\n  //@@ Free the GPU memory here\n  cudaFree(deviceA);\n  cudaFree(deviceB);\n  cudaFree(deviceC);\n  wbTime_stop(GPU, \"Freeing GPU Memory\");\n\n  wbSolution(args, hostC, numARows, numBColumns);\n\n  free(hostA);\n  free(hostB);\n  free(hostC);\n\n  return 0;\n}...",
                "type": "Function"
            }],
            "cform": "__global__ void  sgemm(float * A /* Parameter*/, float * B /* Parameter*/, float * C /* Parameter*/, int  numARows /* Parameter*/, int  numAColumns /* Parameter*/, int  numBRows /* Parameter*/, int  numBColumns){\nint  row = blockIdx.y * blockDim.y + threadIdx.y; /* Declare*/\nint  col = blockIdx.x * blockDim.x + threadIdx.x; /* Declare*/\nif (row < numARows && col < numBColumns){\nfloat  sum = 0; /* Declare*/\nfor (int  ii = 0,ii < numAColumns,++ ii) {\nsum = A[row * numAColumns + ii] * B[ii * numBColumns + col]; /* Assign*/\n}\nC[row * numBColumns + col] = sum; /* Assign*/\n}\n}\nint  main(int  argc /* Parameter*/, char ** argv){\nint  args; /* Declare*/\nfloat * hostA; /* Declare*/\nfloat * hostB; /* Declare*/\nfloat * hostC; /* Declare*/\nfloat * deviceA; /* Declare*/\nfloat * deviceB; /* Declare*/\nfloat * deviceC; /* Declare*/\nint  numARows; /* Declare*/\nint  numAColumns; /* Declare*/\nint  numBRows; /* Declare*/\nint  numBColumns; /* Declare*/\nint  numCRows; /* Declare*/\nint  numCColumns; /* Declare*/\nargs = wbArg_read(argc /* Identifier*/, argv); /* Assign*/\nwbTime_start(\"Generic\" /* String*/, \"Importing data and creating memory on host\"); /* Call*/\nhostA = wbImport(\"input0\" /* String*/, & numARows /* UnaryOperator*/, & numAColumns); /* Assign*/\nhostB = wbImport(\"input1\" /* String*/, & numBRows /* UnaryOperator*/, & numBColumns); /* Assign*/\nhostC = malloc(numARows * numBColumns * sizeof(float )); /* Assign*/\nwbTime_stop(\"Generic\" /* String*/, \"Importing data and creating memory on host\"); /* Call*/\nnumCRows = numARows; /* Assign*/\nnumCColumns = numBColumns; /* Assign*/\nwbLog(\"TRACE\" /* String*/, \"The dimensions of A are \" /* String*/, numARows /* Identifier*/, \" x \" /* String*/, numAColumns); /* Call*/\nwbLog(\"TRACE\" /* String*/, \"The dimensions of B are \" /* String*/, numBRows /* Identifier*/, \" x \" /* String*/, numBColumns); /* Call*/\nwbLog(\"TRACE\" /* String*/, \"The dimensions of C are \" /* String*/, numCRows /* Identifier*/, \" x \" /* String*/, numCColumns); /* Call*/\nwbTime_start(\"GPU\" /* String*/, \"Allocating GPU memory.\"); /* Call*/\nwbTime_stop(\"GPU\" /* String*/, \"Allocating GPU memory.\"); /* Call*/\nwbTime_start(\"GPU\" /* String*/, \"Copying input memory to the GPU.\"); /* Call*/\nwbTime_stop(\"GPU\" /* String*/, \"Copying input memory to the GPU.\"); /* Call*/\nstruct dim3  blockDim = {16 /* Integer32*/, 16}; /* Declare*/\nstruct dim3  gridDim = {ceil((numAColumns) / blockDim.x) /* Call*/, ceil((numBRows) / blockDim.y)}; /* Declare*/\nwbLog(\"TRACE\" /* String*/, \"The block dimensions are \" /* String*/, blockDim.x /* Member*/, \" x \" /* String*/, blockDim.y); /* Call*/\nwbLog(\"TRACE\" /* String*/, \"The grid dimensions are \" /* String*/, gridDim.x /* Member*/, \" x \" /* String*/, gridDim.y); /* Call*/\nwbTime_start(\"Compute\" /* String*/, \"Performing CUDA computation\"); /* Call*/\nsgemm<<<gridDim /* Identifier*/, blockDim>>>(deviceA /* Identifier*/, deviceB /* Identifier*/, deviceC /* Identifier*/, numARows /* Identifier*/, numAColumns /* Identifier*/, numBRows /* Identifier*/, numBColumns); /* Call*/\ncudaDeviceSynchronize(); /* Call*/\nwbTime_stop(\"Compute\" /* String*/, \"Performing CUDA computation\"); /* Call*/\nwbTime_start(\"Copy\" /* String*/, \"Copying output memory to the CPU\"); /* Call*/\nwbTime_stop(\"Copy\" /* String*/, \"Copying output memory to the CPU\"); /* Call*/\nwbTime_start(\"GPU\" /* String*/, \"Freeing GPU Memory\"); /* Call*/\ncudaFree(deviceA); /* Call*/\ncudaFree(deviceB); /* Call*/\ncudaFree(deviceC); /* Call*/\nwbTime_stop(\"GPU\" /* String*/, \"Freeing GPU Memory\"); /* Call*/\nwbSolution(args /* Identifier*/, hostC /* Identifier*/, numARows /* Identifier*/, numBColumns); /* Call*/\nfree(hostA); /* Call*/\nfree(hostB); /* Call*/\nfree(hostC); /* Call*/\nreturn 0; /* Return*/\n}\n",
            "loc": {"end": {"column": 0, "line": 0}, "start": {"column": 0, "line": 0}},
            "raw": "",
            "type": "Program"
        };
    }
}
