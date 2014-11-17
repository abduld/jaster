/// <reference path="../typings/tsd.d.ts" />
/// <reference path="utils.ts" />
/// <reference path="reactutils.ts" />
import React = require("react");
import ReactUtils = require("reactutils");
import _ = require("underscore");

var MAXWIDTH : number = 1024 ;
var MAXHEIGHT : number = 798 ;

export interface ThreadVisualizationProps {
    blockIdx : Core.Dim3;
    blockDim: Core.Dim3;
    gridDim : Core.Dim3;
    threadIdx : Core.Dim3;
    activated: boolean;
}

interface ThreadVisualizationState {
    highlighted: boolean;
}

class ThreadVisualization extends ReactUtils.Component<ThreadVisualizationProps, ThreadVisualizationState> {
    private width : number;
    private height : number;
    private offsetX : number;
    private offsetY : number;

    getInitialState() {
        this.width = 100.0 / (this.props.blockDim.x);
        this.height = 100.0 / (this.props.blockDim.y);
        this.offsetX = this.width * (this.props.threadIdx.x);
        this.offsetY = this.height * (this.props.threadIdx.y);
        return {
            highlighted: false,
            activated: false
        };
    }
    setState(state : ThreadVisualizationState) {
        this.state = state;
    }

    public activate() {
        this.setState({
            highlighted: false,
            activated: true
        });
    }
    highlight() {
        this.setState({
            highlighted: true,
            activated: this.props.activated
        });
    }

    private getFill() : string {
        if (this.props.activated) {
            return "white";
        } else {
            return "black";
        }
    }

    private getStroke() : string {
        if (this.state.highlighted) {
            return "yellow";
        } else {
            return "black";
        }
    }

    render() {
        return React.DOM.rect({
            x: this.offsetX + "%",
            y : this.offsetY + "%",
            width: (this.width * 0.9) + "%",
            height: (this.height * 0.9) + "%",
            fill: this.getFill(),
            stroke: this.getStroke()
        });
    }
}
export var threadVisualization = ReactUtils.createClass<ThreadVisualizationProps, ThreadVisualizationState>(
    React.createClass, ThreadVisualization);

export interface BlockVisualizationProps {
    blockIdx : Core.Dim3;
    blockDim : Core.Dim3;
    gridDim : Core.Dim3;
}

interface BlockVisualizationState {
    highlighted : boolean;
    activated: boolean;
}

class BlockVisualization extends ReactUtils.Component<BlockVisualizationProps, BlockVisualizationState> {
    private width : number;
    private height : number;
    private offsetX : number;
    private offsetY : number;
    private data : React.ReactComponentElement<ThreadVisualizationProps>[][][];
    getInitialState() {
        setTimeout(this.activate, 2*(30 * this.props.blockIdx.x  + 100 * this.props.blockIdx.y + 200 * Math.random())*3);
        this.width = MAXWIDTH / this.props.gridDim.x;
        this.height = MAXHEIGHT / this.props.gridDim.y;
        this.offsetX = this.width * (this.props.blockIdx.x) + 20;
        this.offsetY = this.height * (this.props.blockIdx.y);
        return {
            highlighted: false,
            activated: false
        };
    }

    private makeThreads() : React.ReactComponentElement<ThreadVisualizationProps>[][][] {
        return _.range(this.props.blockDim.z).map((z) => {
            return _.range(this.props.blockDim.y).map((y) => {
                return _.range(this.props.blockDim.x).map((x) => {
                    return React.createElement(threadVisualization, {
                        blockIdx: this.props.blockIdx,
                        activated: this.state.activated,
                        blockDim: this.props.blockDim,
                        gridDim: this.props.gridDim,
                        threadIdx: new Core.Dim3(x, y, z)
                    });
                });
            });
        });
    }
    activate() {
        this.setState({
            highlighted: this.state.highlighted,
            activated: true
        });
    }
    highlight() {
        this.setState({
            highlighted: true,
            activated: this.state.activated
        });
    }
    render() {
        console.log("Rendering...");
        return React.DOM.svg({
            x : this.offsetX,
            y : this.offsetY,
            width: this.width * 0.9,
            height: this.height * 0.85
            //fill: "black"
        }, this.props.children);
    }
}

export var blockVisualization = ReactUtils.createClass<BlockVisualizationProps, BlockVisualizationState>(
    React.createClass, BlockVisualization);

export interface GridVisualizationProps {
    gridDim : Core.Dim3;
    blockDim: Core.Dim3;
}

interface GridVisualizationState {
}

class GridVisualization extends ReactUtils.Component<GridVisualizationProps, GridVisualizationState> {
    private data : React.ReactComponentElement<BlockVisualizationProps>[][][];
    private makeBlocks() : React.ReactComponentElement<BlockVisualizationProps>[][][] {
        return _.range(this.props.gridDim.z).map((z) => {
            return _.range(this.props.gridDim.y).map((y) => {
                return _.range(this.props.gridDim.x).map((x) => {
                    var blockIdx = new (x, y, z);
                    return React.createElement(blockVisualization, {
                            blockIdx: blockIdx,
                            blockDim: this.props.blockDim,
                            gridDim: this.props.gridDim
                        },
                        _.flatten(this.makeThreads(blockIdx))
                    );
                });
            });
        });
    }

    private makeThreads(blockIdx : Core.Dim3) : React.ReactComponentElement<ThreadVisualizationProps>[][][] {
        return _.range(this.props.blockDim.z).map((z) => {
            return _.range(this.props.blockDim.y).map((y) => {
                return _.range(this.props.blockDim.x).map((x) => {
                    return React.createElement(threadVisualization, {
                        blockIdx: blockIdx,
                        activated: false,
                        blockDim: this.props.blockDim,
                        gridDim: this.props.gridDim,
                        threadIdx: new (x, y, z)
                    });
                });
            });
        });
    }
    getInitialState() {
        this.data = this.makeBlocks();
        return { };
    }

    render() {
        return React.DOM.svg({
            xmlns : "http://www.w3.org/2000/svg",
            "xmlns:xlink" : "http://www.w3.org/1999/xlink",
            version: 1.1,
            width: 2 * MAXWIDTH,
            height: 2 * MAXHEIGHT
        }, this.data);
    }
}

export var gridVisualization = createClass<GridVisualizationProps, GridVisualizationState>(
    React.createClass, GridVisualization);
