﻿
module lib {
    export module example {
        export var mp1: any = {
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
                                                            "address_spaces": [

                                                            ],
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
                                                            "address_spaces": [

                                                            ],
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
                                                            "qualifiers": [

                                                            ],
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
                                                            "address_spaces": [

                                                            ],
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
                                                            "address_spaces": [

                                                            ],
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
                                                            "qualifiers": [

                                                            ],
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
                                                        "address_spaces": [

                                                        ],
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
                                                        "address_spaces": [

                                                        ],
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
                                                        "qualifiers": [

                                                        ],
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
                                                            "address_spaces": [

                                                            ],
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
                                                            "qualifiers": [

                                                            ],
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
                                                        "address_spaces": [

                                                        ],
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
                                                        "qualifiers": [

                                                        ],
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
                                                                "address_spaces": [

                                                                ],
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
                                                                "qualifiers": [

                                                                ],
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
                                                            "address_spaces": [

                                                            ],
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
                                                            "qualifiers": [

                                                            ],
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
                                                                "address_spaces": [

                                                                ],
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
                                                                "qualifiers": [

                                                                ],
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
                                                            "address_spaces": [

                                                            ],
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
                                                            "qualifiers": [

                                                            ],
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
                                            "address_spaces": [

                                            ],
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
                                            "qualifiers": [

                                            ],
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
                                            "address_spaces": [

                                            ],
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
                                            "qualifiers": [

                                            ],
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
                            "type": "ParameterExpression"
                        }
                    ],
                    "raw": "",
                    "type": "Function"
                },
                {
                    "attributes": [

                    ],
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
                                        "address_spaces": [

                                        ],
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
                                        "qualifiers": [

                                        ],
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
                                                "address_spaces": [

                                                ],
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
                                                "qualifiers": [

                                                ],
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
                                                        "address_spaces": [

                                                        ],
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
                                                        "qualifiers": [

                                                        ],
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
                            {

                            },
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
                                            "address_spaces": [

                                            ],
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
                                            "qualifiers": [

                                            ],
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
                                                    "address_spaces": [

                                                    ],
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
                                                    "qualifiers": [

                                                    ],
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
                                            "address_spaces": [

                                            ],
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
                                            "qualifiers": [

                                            ],
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
                                                    "address_spaces": [

                                                    ],
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
                                                    "qualifiers": [

                                                    ],
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
                                            "address_spaces": [

                                            ],
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
                                            "qualifiers": [

                                            ],
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
                                                    "address_spaces": [

                                                    ],
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
                                                    "qualifiers": [

                                                    ],
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
                                                        "address_spaces": [

                                                        ],
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
                                                        "qualifiers": [

                                                        ],
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
                            {

                            },
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
                                            "address_spaces": [

                                            ],
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
                                            "qualifiers": [

                                            ],
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
                            {

                            },
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
                                                        "address_spaces": [

                                                        ],
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
                                                        "qualifiers": [

                                                        ],
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
                                                    "address_spaces": [

                                                    ],
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
                                                    "qualifiers": [

                                                    ],
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
                            {

                            },
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
                            {

                            },
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
                                                    "address_spaces": [

                                                    ],
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
                                                    "qualifiers": [

                                                    ],
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
                                            "address_spaces": [

                                            ],
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
                                            "qualifiers": [

                                            ],
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
                                                    "address_spaces": [

                                                    ],
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
                                                    "qualifiers": [

                                                    ],
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
                                            "address_spaces": [

                                            ],
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
                                            "qualifiers": [

                                            ],
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
                                                    "address_spaces": [

                                                    ],
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
                                                    "qualifiers": [

                                                    ],
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
                                            "address_spaces": [

                                            ],
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
                                            "qualifiers": [

                                            ],
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
                            {

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
                                                "address_spaces": [

                                                ],
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
                                                "qualifiers": [

                                                ],
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
                                                "address_spaces": [

                                                ],
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
                                                "qualifiers": [

                                                ],
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
                                            "address_spaces": [

                                            ],
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
                                            "qualifiers": [

                                            ],
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
                                            "address_spaces": [

                                            ],
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
                                                "address_spaces": [

                                                ],
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
                                                "qualifiers": [

                                                ],
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
                                                "address_spaces": [

                                                ],
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
                                                "qualifiers": [

                                                ],
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
                                            "address_spaces": [

                                            ],
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
                                            "qualifiers": [

                                            ],
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
                                            "address_spaces": [

                                            ],
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
                            {

                            },
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
                                                        "address_spaces": [

                                                        ],
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
                                                        "qualifiers": [

                                                        ],
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
                                                        "address_spaces": [

                                                        ],
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
                                                        "qualifiers": [

                                                        ],
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
                                                                    "address_spaces": [

                                                                    ],
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
                                                                    "qualifiers": [

                                                                    ],
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
                                                                    "address_spaces": [

                                                                    ],
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
                                                                    "qualifiers": [

                                                                    ],
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
                                                "address_spaces": [

                                                ],
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
                                                "qualifiers": [

                                                ],
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
                                                "address_spaces": [

                                                ],
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
                                                "qualifiers": [

                                                ],
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
                                                "address_spaces": [

                                                ],
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
                                                "qualifiers": [

                                                ],
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
                                            "address_spaces": [

                                            ],
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
                                            "qualifiers": [

                                            ],
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
                                                    "address_spaces": [

                                                    ],
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
                                                    "qualifiers": [

                                                    ],
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
                                                    "address_spaces": [

                                                    ],
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
                                                    "qualifiers": [

                                                    ],
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
                                "arguments": [

                                ],
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
                            {

                            },
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
                            {

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
                                                "address_spaces": [

                                                ],
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
                                                "qualifiers": [

                                                ],
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
                                                "address_spaces": [

                                                ],
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
                                                "qualifiers": [

                                                ],
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
                                            "address_spaces": [

                                            ],
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
                                            "qualifiers": [

                                            ],
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
                                            "address_spaces": [

                                            ],
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
                            {

                            },
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
                            {

                            },
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
                            {

                            },
                            {
                                "arguments": [
                                    {
                                        "cform": "args",
                                        "kind": {
                                            "address_spaces": [

                                            ],
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
                                            "qualifiers": [

                                            ],
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
                                                "address_spaces": [

                                                ],
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
                                                "qualifiers": [

                                                ],
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
                                            "address_spaces": [

                                            ],
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
                                            "qualifiers": [

                                            ],
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
                                                "address_spaces": [

                                                ],
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
                                                "qualifiers": [

                                                ],
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
                                                "address_spaces": [

                                                ],
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
                                                "qualifiers": [

                                                ],
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
                                                "address_spaces": [

                                                ],
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
                                                "qualifiers": [

                                                ],
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
        };
    }
}