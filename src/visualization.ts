/// <reference path="../typings/tsd.d.ts" />
/// <reference path="utils.ts" />

import React = require("react");
import ReactUtils = require("./react_utils");
import Utils = require("./utils");
import _ = require("underscore");

var MAXWIDTH : number = 75;
var MAXHEIGHT : number = 100;

export interface ThreadVisualizationProps {
    blockIdx : Utils.Dim3;
    blockDim: Utils.Dim3;
    gridDim : Utils.Dim3;
    threadIdx : Utils.Dim3;
}

interface ThreadVisualizationState {
    highlighted: boolean;
    activated : boolean;
}

class ThreadVisualization extends ReactUtils.Component<ThreadVisualizationProps, ThreadVisualizationState> {
    private width : number;
    private height : number;
    private offsetX : number;
    private offsetY : number;

    getInitialState() {

        this.width = MAXWIDTH / (this.props.gridDim.x * this.props.blockDim.x);
        this.height = MAXHEIGHT / (this.props.gridDim.y * this.props.blockDim.y);
        this.offsetX = this.width * (this.props.gridDim.x * this.props.blockIdx.x + this.props.threadIdx.x);
        this.offsetY = this.height * (this.props.gridDim.y * this.props.blockIdx.y + this.props.threadIdx.y);
        return {
            highlighted: false,
            activated: false
        };
    }
    setState(state : ThreadVisualizationState) {
        this.state = state;
    }

    activate() {
        this.setState({
            highlighted: false,
            activated: true
        });
    }
    highlight() {
        this.setState({
            highlighted: true,
            activated: this.state.activated
        });
    }

    private getFill() : string {
        if (this.state.activated) {
            return "#fff";
        } else {
            return "#000";
        }
    }
    render() {
        return React.DOM.rect({
            x: this.offsetX + "%",
            y : this.offsetY + "%",
            width: (this.width * 0.8) + "%",
            height: (this.height * 0.8) + "%",
            fill: this.getFill()
        });
    }
}
export var cell = ReactUtils.createClass<ThreadVisualizationProps, ThreadVisualizationState>(
    React.createClass, ThreadVisualization);

export interface BlockVisualizationProps {
    blockIdx : Utils.Dim3;
    blockDim : Utils.Dim3;
    gridDim : Utils.Dim3;
}

interface BlockVisualizationState {
    highlighted : boolean;
}

class BlockVisualization extends ReactUtils.Component<BlockVisualizationProps, BlockVisualizationState> {
    private width : number;
    private height : number;
    private offsetX : number;
    private offsetY : number;
    private data : React.ReactComponentElement<ThreadVisualizationProps>[][][];
    private makeCells() : React.ReactComponentElement<ThreadVisualizationProps>[][][] {
        return _.range(this.props.blockDim.z).map((z) => {
            return _.range(this.props.blockDim.y).map((y) => {
                return _.range(this.props.blockDim.x).map((x) => {
                    return React.createElement(cell, {
                        blockIdx: this.props.blockIdx,
                        blockDim: this.props.blockDim,
                        gridDim: this.props.gridDim,
                        threadIdx: new Utils.Dim3(x, y, z)
                    });
                });
            });
        });
    }
    getInitialState() {
        this.width = MAXWIDTH / this.props.gridDim.x;
        this.height = MAXHEIGHT / this.props.gridDim.y;
        this.offsetX = this.width * (this.props.blockIdx.x);
        this.offsetY = this.height * (this.props.blockIdx.y);
        this.data = this.makeCells();
        return {
            highlighted: false
        };
    }

    highlight() {
        /*
        _.flatten(this.data).forEach(function(c : React.ReactComponentElement<ThreadVisualizationProps>) {
            c.highlight();
        });
        */
        this.setState({
            highlighted: true
        });
    }
    render() {
        return React.DOM.svg({
            //x: this.offsetX,
            //y : this.offsetY,
            //width: this.width * 0.8,
            //height: this.height * 0.8,
            //fill: "black"
        }, this.data);
    }
}

export var blockVisualization = ReactUtils.createClass<BlockVisualizationProps, BlockVisualizationState>(
    React.createClass, BlockVisualization);

export interface GridVisualizationProps {
    gridDim : Utils.Dim3;
    blockDim: Utils.Dim3;
}

interface GridVisualizationState {
}

class GridVisualization extends ReactUtils.Component<GridVisualizationProps, GridVisualizationState> {
    private data : React.ReactComponentElement<BlockVisualizationProps>[][][];
    private makeBlocks() : React.ReactComponentElement<BlockVisualizationProps>[][][] {
        return _.range(this.props.gridDim.z).map((z) => {
            return _.range(this.props.gridDim.y).map((y) => {
                return _.range(this.props.gridDim.x).map((x) => {
                    return React.createElement(blockVisualization, {
                        blockIdx: new Utils.Dim3(x, y, z),
                        blockDim: this.props.blockDim,
                        gridDim: this.props.gridDim
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
            version: 1.1
        }, this.data);
    }
}

export var gridVisualization = ReactUtils.createClass<GridVisualizationProps, GridVisualizationState>(
    React.createClass, GridVisualization);
