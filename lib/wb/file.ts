module lib.wb {
    export function wbImport(state, stack, file: string, numRowsRef, numColRef?): number[] {
        switch (state.mpnum) {
            case 1:
                var size = 1000;
                if (_.isObject(numRowsRef)) {
                    numRowsRef.stack[numRowsRef.id] = size;
                }
                return _.range(size);
            case 2:

                var rows = 64;
                var cols = 64;
                if (_.isObject(numRowsRef)) {
                    numRowsRef.stack[numRowsRef.id] = rows;
                }
                if (_.isObject(numColRef)) {
                    numColRef.stack[numColRef.id] = cols;
                }
                return lib.utils.castTo<number[]>(_.flatten(_.map(_.range(rows), () => _.range(cols))));
        }
        return [];
    }
}
