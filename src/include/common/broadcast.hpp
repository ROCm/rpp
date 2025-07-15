#include "rpp.h"

inline void checkEqualBatchSize(RpptGenericDescPtr src1DescriptorPtrND, RpptGenericDescPtr src2DescriptorPtrND) {
    if(src1DescriptorPtrND->dims[0] != src2DescriptorPtrND->dims[0]) {
        printf("Batch Size of Inputs must be equal\n");
        exit(0);
    }
}

inline Rpp32s getMaxNumDims(RpptGenericDescPtr src1DescriptorPtrND, RpptGenericDescPtr src2DescriptorPtrND) {
    if(src1DescriptorPtrND->numDims > src2DescriptorPtrND->numDims)
        return src1DescriptorPtrND->numDims;
    return src2DescriptorPtrND->numDims;
}

inline void UpdateDstDimension(RpptGenericDescPtr srcDescriptorPtr, RpptGenericDescPtr dstDescriptorPtr, Rpp32s idx) {
    int srcDimension;
    int src_ndim = srcDescriptorPtr->numDims;
    int dst_ndim = dstDescriptorPtr->numDims;
    if(idx >= src_ndim - 1)
        srcDimension = 1;
    else
        srcDimension = srcDescriptorPtr->dims[src_ndim - 1 - idx];
    if (srcDimension != 1 && srcDimension != dstDescriptorPtr->dims[dst_ndim - 1 - idx])
        if (dstDescriptorPtr->dims[dst_ndim - 1 - idx] != 1) {
            printf("Incompatible dimension and leads to broadcasting failure\n");
            exit(0);
        }
    if (srcDimension != dstDescriptorPtr->dims[dst_ndim - 1 - idx] && dstDescriptorPtr->dims[dst_ndim - 1 - idx] == 1)
        dstDescriptorPtr->dims[dst_ndim - 1 - idx] = srcDimension;
}

inline void CopyDescriptorForBroadcasting(RpptGenericDescPtr descPtrND, RpptGenericDescPtr broadcastDescPtrND) {
    RpptGenericDesc descND = *descPtrND;
    RpptGenericDesc broadcastDescND = descND;
    broadcastDescPtrND = &broadcastDescND;
}

inline void BroadcastDstShape(RpptGenericDescPtr src1DescriptorPtrND, RpptGenericDescPtr src2DescriptorPtrND, RpptGenericDescPtr dstDescriptorPtrND) {
    int ndim = getMaxNumDims(src1DescriptorPtrND, src2DescriptorPtrND);
    dstDescriptorPtrND->numDims = ndim;
    dstDescriptorPtrND->dims[0] = src1DescriptorPtrND->dims[0];
    for(int i = 0; i < ndim - 1; i++) {
        dstDescriptorPtrND->dims[ndim - 1 - i] = 1;
        UpdateDstDimension(src1DescriptorPtrND, dstDescriptorPtrND, i);
        UpdateDstDimension(src2DescriptorPtrND, dstDescriptorPtrND, i);
    }
}

inline int GetShapeAtIndex(RpptGenericDescPtr descriptorPtrND, int index, int ndim) {
    index -= ndim - descriptorPtrND->numDims;
    if(index < 1)
        return 1;
    return descriptorPtrND->dims[index];
}

// In DALI they check for both inputs and outputs - RpptGenericDescPtr src1DescriptorPtrND, RpptGenericDescPtr src2DescriptorPtrND,
// But ideally checking dst Shape is enough 
inline bool SkipIndexForBroadcasting(RpptGenericDescPtr dstDescriptorPtrND, int index) {
    if(GetShapeAtIndex(dstDescriptorPtrND, index, dstDescriptorPtrND->numDims) != 1)
        return false;
    return true;
}

inline bool CanCollapse(int src1StartShape, int src2StartShape, int src1IndexShape, int src2IndexShape) {
    bool src1IndexFlag = src1IndexShape == 1;
    bool src1StartFlag = src1StartShape == 1;
    if(src1IndexFlag != src1StartFlag)
        return false;
    bool src2IndexFlag = src2IndexShape == 1;
    bool src2StartFlag = src2StartShape == 1;
    if(src2IndexFlag != src2StartFlag)
        return false;
    return true;
}

inline void AddGroupDimension(RpptGenericDescPtr src1DescriptorPtrND, RpptGenericDescPtr src2DescriptorPtrND, RpptGenericDescPtr dstDescriptorPtrND, int* volumes, int index, int n_groups, std::vector<int>& updated_dims) {
    int ndim = dstDescriptorPtrND->numDims;
    updated_dims.push_back(volumes[0]);
    updated_dims.push_back(volumes[1]);
    updated_dims.push_back(volumes[2]);
    volumes[0] = GetShapeAtIndex(src1DescriptorPtrND, index, ndim);
    volumes[1] = GetShapeAtIndex(src2DescriptorPtrND, index, ndim);
    volumes[2] = GetShapeAtIndex(dstDescriptorPtrND, index, ndim);
}

inline void GroupShapes(RpptGenericDescPtr src1DescriptorPtrND, RpptGenericDescPtr src2DescriptorPtrND, RpptGenericDescPtr dstDescriptorPtrND) {
    int ndim = dstDescriptorPtrND->numDims;
    int d = 1;
    int n_groups = 1;

    int volumes[3];
    for(int i = 0; i < 3; i++)
        volumes[i] = 1;
    std::vector<int> updated_dims;
    updated_dims.reserve(ndim * 3);

    for(; d < ndim; d++) {
        if (!SkipIndexForBroadcasting(dstDescriptorPtrND, d))
            break;
    }

    if(d < ndim) {
        int src1StartShape = GetShapeAtIndex(src1DescriptorPtrND, d, ndim);
        int src2StartShape = GetShapeAtIndex(src2DescriptorPtrND, d, ndim);
        int dstStartShape = GetShapeAtIndex(dstDescriptorPtrND, d, ndim);
        volumes[0] *= src1StartShape;
        volumes[1] *= src2StartShape;
        volumes[2] *= dstStartShape;

        for (d++; d < ndim; d++) {
            int dstIndexShape = GetShapeAtIndex(dstDescriptorPtrND, d, ndim);
            if(dstIndexShape == 1)
                continue;
            int src1IndexShape = GetShapeAtIndex(src1DescriptorPtrND, d, ndim);
            int src2IndexShape = GetShapeAtIndex(src2DescriptorPtrND, d, ndim);

            if (CanCollapse(src1StartShape, src2StartShape, src1IndexShape, src2IndexShape)) {
                volumes[0] *= src1IndexShape;
                volumes[1] *= src2IndexShape;
                volumes[2] *= dstIndexShape;
            }
            else {
                updated_dims.push_back(volumes[0]);
                updated_dims.push_back(volumes[1]);
                updated_dims.push_back(volumes[2]);
                volumes[0] = src1StartShape = src1IndexShape;
                volumes[1] = src2StartShape = src2IndexShape;
                volumes[2] = dstStartShape = dstIndexShape;
                n_groups++;
            }
        }
        updated_dims.push_back(volumes[0]);
        updated_dims.push_back(volumes[1]);
        updated_dims.push_back(volumes[2]);
        n_groups++;
        src1DescriptorPtrND->numDims = n_groups;
        src2DescriptorPtrND->numDims = n_groups;
        dstDescriptorPtrND->numDims = n_groups;

        int idx = 0;
        for(d = 1; d < n_groups; d++) {
            src1DescriptorPtrND->dims[d] = updated_dims[idx++];
            src2DescriptorPtrND->dims[d] = updated_dims[idx++];
            dstDescriptorPtrND->dims[d] = updated_dims[idx++];
        }

        compute_strides(src1DescriptorPtrND->strides, src1DescriptorPtrND->dims, src1DescriptorPtrND->numDims);
        compute_strides(src2DescriptorPtrND->strides, src2DescriptorPtrND->dims, src2DescriptorPtrND->numDims);
        compute_strides(dstDescriptorPtrND->strides, dstDescriptorPtrND->dims, dstDescriptorPtrND->numDims);
    }
}

inline void StridesForBroadcasting(RpptGenericDescPtr srcDescriptorPtrND, RpptGenericDescPtr dstDescriptorPtrND) {
    int ndim = dstDescriptorPtrND->numDims;
    for(int i = 0; i < ndim - 1; i++) {
        if((srcDescriptorPtrND->dims[ndim - 1 - i] != dstDescriptorPtrND->dims[ndim - 1 - i]) && (srcDescriptorPtrND->dims[ndim - 1 - i] == 1)) {
            srcDescriptorPtrND->strides[ndim - 1 - i] = 0;
        }
    }
}
