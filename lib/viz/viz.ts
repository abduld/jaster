/// <reference path="./reactutils.ts" />

/// <reference path="react.d.ts" />

module lib {
    export module viz {


        import utils = lib.utils;
        import Dim3 = lib.cuda.Dim3;

        var MAXWIDTH: number = 1024;
        var MAXHEIGHT: number = 798;

        export interface ThreadVisualizationProps extends React.ReactAttributes {
            blockIdx: Dim3;
            blockDim: Dim3;
            gridDim: Dim3;
            threadIdx: Dim3;
            activated: boolean;
            key? : string;
            id? : string;
            options: number;
        }

        interface ThreadVisualizationState {
            highlighted: boolean;
            activated: boolean;
        }

        class ThreadVisualization extends Component<ThreadVisualizationProps, ThreadVisualizationState> {
            private width: number;
            private height: number;
            private offsetX: number;
            private offsetY: number;
            private tooltip_;

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


            setState(state: ThreadVisualizationState) {
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
                    activated: this.props.activated
                });
            }

            private getFill(): string {
              if (this.props.options === 1 && this.state.activated && this.props.blockIdx.x === 0 &&
                this.props.threadIdx.x == 0) {
                return "red";
              }
                if (this.state.activated) {
                  return "#7a62d3";
                } else {
                    return "#ccc";
                }
            }

            private getStroke(): string {
                if (this.state.highlighted) {
                  return "#7a62d3";
                } else {
                    return "#ccc";
                }
            }

            componentWillReceiveProps(props){
              //console.log(props.id);
              if (this.props.activated != props.activated) {
                setTimeout(this.activate, _.parseInt(this.props.id) *  Math.random());
              }
            }
            shouldComponentUpdate(props, state) {
              return this.state.activated != state.activated;
            }

 onMouseMove(evt)
{
  console.log("Mouse move...");
  if (_.isUndefined(this.tooltip_)) {
  var svgDocument = evt.target.ownerDocument;
  this.tooltip_ = svgDocument.getElementById('tooltip');
}
console.log([evt.clientX, evt.clientY]);
this.tooltip_.setAttributeNS(null,"x",evt.clientX - 140);
  this.tooltip_.setAttributeNS(null,"visibility","visible");
  this.tooltip_.setAttributeNS(null,"y",evt.clientY - 90);
  this.tooltip_.firstChild.data = this.props.threadIdx.toString();
}

 onMouseOut()
{
  if (_.isUndefined(this.tooltip_)) {
    return ;
  }
  this.tooltip_.setAttributeNS(null,"visibility","hidden");

}
            render() {
                return React.DOM.rect({
                    x: this.offsetX + "%",
                    y: this.offsetY + "%",
                    width: (this.width * 0.9) + "%",
                    height: (this.height * 0.9) + "%",
                    fill: this.getFill(),
                    onClick: this.onMouseMove,
                    onMouseMove: this.onMouseMove,
                    onMouseOver: this.onMouseMove,
                    onMouseOut: this.onMouseOut,
                    stroke: this.getStroke()
                });
            }
        }
        export var threadVisualizationClass = createClass(ThreadVisualization);
        export var threadVisualization = React.createFactory<ThreadVisualizationProps>(threadVisualizationClass);

        export interface BlockVisualizationProps extends React.ReactAttributes  {
            blockIdx: Dim3;
            blockDim: Dim3;
            gridDim: Dim3;
            options: number;
            children? : React.ComponentElement<ThreadVisualizationProps>[];
        }

        interface BlockVisualizationState {
            highlighted: boolean;
            activated: boolean;
        }

        class BlockVisualization extends Component<BlockVisualizationProps, BlockVisualizationState> {
            private width: number;
            private height: number;
            private offsetX: number;
            private offsetY: number;
            private data: React.ComponentElement<ThreadVisualizationProps>[][][];

            getInitialState() {
                setTimeout(this.activate, 4 * (30 * this.props.blockIdx.x + 100 * this.props.blockIdx.y + 200 * Math.random()) * 3);
                this.width = MAXWIDTH / this.props.gridDim.x;
                this.height = MAXHEIGHT / this.props.gridDim.y;
                this.offsetX = this.width * (this.props.blockIdx.x) + 20;
                this.offsetY = this.height * (this.props.blockIdx.y);
                return {
                    highlighted: false,
                    activated: false
                };
            }

            private makeThreads(): React.ComponentElement<ThreadVisualizationProps>[][][] {
                return _.range(this.props.blockDim.z).map((z) => {
                    return _.range(this.props.blockDim.y).map((y) => {
                        return _.range(this.props.blockDim.x).map((x) => {
                          var blockIdx=  this.props.blockIdx;
                          var blockDim=  this.props.blockDim;
                          var gridDim = this.props.gridDim;
                          var threadIdx = new Dim3(x, y, z);
                          var key = lib.utils._toString(
                            (
                              (
                                blockIdx.z * blockDim.z + threadIdx.z
                                )*gridDim.y +
                                (blockIdx.y * blockDim.y + threadIdx.y)
                                )*gridDim.x +
                                (blockIdx.x * blockDim.x + threadIdx.x)
                                ).toString();
                                debugger;
                          return threadVisualization({
                            blockIdx: blockIdx,
                            activated: false,
                            blockDim: blockDim,
                            gridDim: gridDim,
                            threadIdx: threadIdx,
                            options: this.props.options,
                            key: key,
                            id: key
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
              var self = this;
                return React.DOM.svg({
                    x: this.offsetX,
                    y: this.offsetY,
                    width: this.width * 0.9,
                    height: this.height * 0.85,
                    fill: "#ccc"
                }, React.Children.map(this.props.children, function(child: React.ComponentElement<ThreadVisualizationProps>) {
            return React.addons.cloneWithProps(child, { activated: self.state.activated })
        }));
            }
        }

        export var blockVisualizationClass = createClass(BlockVisualization);
        export var blockVisualization = React.createFactory<BlockVisualizationProps>(blockVisualizationClass);

        export interface GridVisualizationProps extends React.ReactAttributes  {
            gridDim: Dim3;
            blockDim: Dim3;
            options: any;
        }

        interface GridVisualizationState {
        }

        class GridVisualization extends Component<GridVisualizationProps, GridVisualizationState> {
            private data: React.ComponentElement<BlockVisualizationProps>[][][];

            private makeBlocks(): React.ComponentElement<BlockVisualizationProps>[][][] {
                return _.range(this.props.gridDim.z).map((z) => {
                    return _.range(this.props.gridDim.y).map((y) => {
                        return _.range(this.props.gridDim.x).map((x) => {
                            var blockIdx = new Dim3(x, y, z);
                            return blockVisualization({
                              options: this.props.options,
                                blockIdx: blockIdx,
                                blockDim: this.props.blockDim,
                                gridDim: this.props.gridDim,
                                id: blockIdx.flattenedLength()
                            },
                            utils.castTo<React.ComponentElement<ThreadVisualizationProps>[]>(_.flatten(this.makeThreads(blockIdx)))
                                );
                        });
                    });
                });
            }

            private makeThreads(blockIdx: Dim3): React.ComponentElement<ThreadVisualizationProps>[][][] {
                return _.range(this.props.blockDim.z).map((z) => {
                    return _.range(this.props.blockDim.y).map((y) => {
                        return _.range(this.props.blockDim.x).map((x) => {
                          var blockDim=  this.props.blockDim;
                          var gridDim = this.props.gridDim;
                          var threadIdx = new Dim3(x, y, z);
                          var key = (
                            (
                              (
                                blockIdx.z * blockDim.z + threadIdx.z
                                )*gridDim.y +
                                (blockIdx.y * blockDim.y + threadIdx.y)
                                )*gridDim.x +
                                (blockIdx.x * blockDim.x + threadIdx.x)
                                ).toString();
                            return threadVisualization({
                                blockIdx: blockIdx,
                                activated: false,
                                blockDim: blockDim,
                                gridDim: gridDim,
                                threadIdx: threadIdx,
                                options: this.props.options,
                                key: key,
                                id: key
                            });
                        });
                    });
                });
            }

            getInitialState() {
                this.data = this.makeBlocks();
                return {};
            }

            render() {
              var data :any[]= this.data;
              data = data.concat([
                React.DOM.text({
                  id: "tooltip",
                  x: 0,
                  y: 0,
                  key: lib.utils.guuid()
                  }, "data")]
                  );
                return React.DOM.svg({
                    xmlns: "http://www.w3.org/2000/svg",
                    "xmlns:xlink": "http://www.w3.org/1999/xlink",
                    version: "1.1",
                    width: 2 * MAXWIDTH,
                    height: 2 * MAXHEIGHT
                }, data);
            }
        }



      export var gridVisualizationClass = createClass(GridVisualization);
      export var gridVisualization = React.createFactory<GridVisualizationProps>(gridVisualizationClass);

    }
}
