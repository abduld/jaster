
/// <reference path="../../ref.ts" />
module Parallel {
	export class WorkerPool {
	private num_workers : number;
	private workers : Worker[];
	constructor(num_workers : number) {
		this.num_workers = num_workers;

	}
}
}