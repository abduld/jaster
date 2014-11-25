var __extends = this.__extends || function (d, b) {
    for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p];
    function __() { this.constructor = d; }
    __.prototype = b.prototype;
    d.prototype = new __();
};
define(["require", "exports", "react", "react/addons", "./reactutils", "underscore", "./../utils/utils"], function(require, exports, React, ReactAddons, ReactUtils, _, utils) {
    var MAXWIDTH = 1024;
    var MAXHEIGHT = 798;

    var ThreadVisualization = (function (_super) {
        __extends(ThreadVisualization, _super);
        function ThreadVisualization() {
            _super.apply(this, arguments);
        }
        ThreadVisualization.prototype.getInitialState = function () {
            this.width = 100.0 / (this.props.blockDim.x);
            this.height = 100.0 / (this.props.blockDim.y);
            this.offsetX = this.width * (this.props.threadIdx.x);
            this.offsetY = this.height * (this.props.threadIdx.y);
            return {
                highlighted: false,
                activated: false
            };
        };
        ThreadVisualization.prototype.setState = function (state) {
            this.state = state;
        };

        ThreadVisualization.prototype.activate = function () {
            this.setState({
                highlighted: false,
                activated: true
            });
        };
        ThreadVisualization.prototype.highlight = function () {
            this.setState({
                highlighted: true,
                activated: this.props.activated
            });
        };

        ThreadVisualization.prototype.getFill = function () {
            if (this.props.activated) {
                return "white";
            } else {
                return "black";
            }
        };

        ThreadVisualization.prototype.getStroke = function () {
            if (this.state.highlighted) {
                return "yellow";
            } else {
                return "black";
            }
        };

        ThreadVisualization.prototype.render = function () {
            return React.DOM.rect({
                x: this.offsetX + "%",
                y: this.offsetY + "%",
                width: (this.width * 0.9) + "%",
                height: (this.height * 0.9) + "%",
                fill: this.getFill(),
                stroke: this.getStroke()
            });
        };
        return ThreadVisualization;
    })(ReactUtils.Component);
    exports.threadVisualization = ReactUtils.createClass(React.createClass, ThreadVisualization);

    var BlockVisualization = (function (_super) {
        __extends(BlockVisualization, _super);
        function BlockVisualization() {
            _super.apply(this, arguments);
        }
        BlockVisualization.prototype.getInitialState = function () {
            setTimeout(this.activate, 2 * (30 * this.props.blockIdx.x + 100 * this.props.blockIdx.y + 200 * Math.random()) * 3);
            this.width = MAXWIDTH / this.props.gridDim.x;
            this.height = MAXHEIGHT / this.props.gridDim.y;
            this.offsetX = this.width * (this.props.blockIdx.x) + 20;
            this.offsetY = this.height * (this.props.blockIdx.y);
            return {
                highlighted: false,
                activated: false
            };
        };

        BlockVisualization.prototype.makeThreads = function () {
            var _this = this;
            return _.range(this.props.blockDim.z).map(function (z) {
                return _.range(_this.props.blockDim.y).map(function (y) {
                    return _.range(_this.props.blockDim.x).map(function (x) {
                        return React.createElement(exports.threadVisualization, {
                            blockIdx: _this.props.blockIdx,
                            activated: _this.state.activated,
                            blockDim: _this.props.blockDim,
                            gridDim: _this.props.gridDim,
                            threadIdx: new utils.Dim3(x, y, z)
                        });
                    });
                });
            });
        };
        BlockVisualization.prototype.activate = function () {
            this.setState({
                highlighted: this.state.highlighted,
                activated: true
            });
        };
        BlockVisualization.prototype.highlight = function () {
            this.setState({
                highlighted: true,
                activated: this.state.activated
            });
        };
        BlockVisualization.prototype.render = function () {
            var children = this.props.children;
            console.log("Rendering...");
            return React.DOM.svg({
                x: this.offsetX,
                y: this.offsetY,
                width: this.width * 0.9,
                height: this.height * 0.85
            }, React.Children.map(children, function (child) {
                return ReactAddons.addons.cloneWithProps(child, { activated: true });
            }.bind(this)));
        };
        return BlockVisualization;
    })(ReactUtils.Component);

    exports.blockVisualization = ReactUtils.createClass(React.createClass, BlockVisualization);

    var GridVisualization = (function (_super) {
        __extends(GridVisualization, _super);
        function GridVisualization() {
            _super.apply(this, arguments);
        }
        GridVisualization.prototype.makeBlocks = function () {
            var _this = this;
            return _.range(this.props.gridDim.z).map(function (z) {
                return _.range(_this.props.gridDim.y).map(function (y) {
                    return _.range(_this.props.gridDim.x).map(function (x) {
                        var blockIdx = new utils.Dim3(x, y, z);
                        return React.createElement(exports.blockVisualization, {
                            blockIdx: blockIdx,
                            blockDim: _this.props.blockDim,
                            gridDim: _this.props.gridDim,
                            children: _.flatten(_this.makeThreads(blockIdx))
                        });
                    });
                });
            });
        };

        GridVisualization.prototype.makeThreads = function (blockIdx) {
            var _this = this;
            return _.range(this.props.blockDim.z).map(function (z) {
                return _.range(_this.props.blockDim.y).map(function (y) {
                    return _.range(_this.props.blockDim.x).map(function (x) {
                        return React.createElement(exports.threadVisualization, {
                            blockIdx: blockIdx,
                            activated: false,
                            blockDim: _this.props.blockDim,
                            gridDim: _this.props.gridDim,
                            threadIdx: new utils.Dim3(x, y, z)
                        });
                    });
                });
            });
        };
        GridVisualization.prototype.getInitialState = function () {
            this.data = this.makeBlocks();
            return {};
        };

        GridVisualization.prototype.render = function () {
            return React.DOM.svg({
                xmlns: "http://www.w3.org/2000/svg",
                "xmlns:xlink": "http://www.w3.org/1999/xlink",
                version: 1.1,
                width: 2 * MAXWIDTH,
                height: 2 * MAXHEIGHT
            }, this.data);
        };
        return GridVisualization;
    })(ReactUtils.Component);

    exports.gridVisualization = ReactUtils.createClass(React.createClass, GridVisualization);
});
//# sourceMappingURL=visualization.js.map
