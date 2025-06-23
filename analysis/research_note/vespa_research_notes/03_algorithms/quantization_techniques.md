# Vespa Quantization Techniques Analysis

## Overview

Vespa supports vector quantization to reduce memory footprint and improve search performance. The core C++ engine provides highly optimized, production-grade support for **INT8 (scalar)** and **binary** quantization, which are deeply integrated with its hardware-accelerated SIMD framework.

This document analyzes both the quantization methods found in the Vespa codebase and other advanced, related concepts like Product Quantization (PQ) and Iterative Quantization (ITQ).

---

## 1. Implemented Quantization Methods in Vespa Core

The following methods are confirmed to be implemented in Vespa's C++ `searchlib` and are optimized for performance.

### 1.1. INT8 Quantization

Vespa provides native support for 8-bit integer vectors through its `Int8Float` cell type. This is the primary method for scalar quantization.

- **Storage**: Vectors are stored as `int8_t` arrays.
- **Performance**: Distance calculations (Euclidean, dot product, prenormalized angular) on `int8_t` vectors are heavily optimized using SIMD instructions (AVX2, AVX-512) via a runtime-dispatching hardware acceleration layer.
- **Conversion**: The application or a feeding component is responsible for converting `float` vectors to `int8_t` before they are fed into Vespa.

**Codebase Evidence:**

The system uses `Int8Float` as a distinct cell type, and the `DistanceFunctionFactory` dispatches to specialized, hardware-accelerated distance functions for it.

```cpp
// From: searchlib/src/vespa/searchlib/tensor/distance_function_factory.cpp

std::unique_ptr<DistanceFunctionFactory>
make_distance_function_factory(DistanceMetric metric, CellType cell_type)
{
    switch (metric) {
        case DistanceMetric::Euclidean:
            switch (cell_type) {
                case CellType::DOUBLE:   return std::make_unique<EuclideanDistanceFunctionFactory<double>>(true);
                case CellType::INT8:     return std::make_unique<EuclideanDistanceFunctionFactory<Int8Float>>(true);
                // ...
            }
        // ... similar cases for Dotproduct, PrenormalizedAngular
    }
}
```

These factories then use the hardware acceleration layer, which has optimized C++ implementations for different CPU architectures.

```cpp
// From: vespalib/src/vespa/vespalib/hwaccelerated/avx2.cpp

double
Avx2Accelerator::squaredEuclideanDistance(const int8_t * a, const int8_t * b, size_t sz) const noexcept {
    return helper::squaredEuclideanDistance(a, b, sz);
}

int64_t
Avx2Accelerator::dotProduct(const int8_t * a, const int8_t * b, size_t sz) const noexcept
{
    return helper::multiplyAdd(a, b, sz);
}
```

### 1.2. Binary Quantization (for Hamming Distance)

Vespa supports binary vectors for use with the `hamming` distance metric. This is implemented using the same `int8_t` cell type, where each byte represents a chunk of 8 bits.

- **Storage**: Binary vectors are packed into `int8_t` arrays.
- **Distance Metric**: The `hamming` distance metric triggers specialized functions that compute the bitwise Hamming distance.
- **Performance**: Hamming distance is optimized using `popcnt` instructions, with AVX-512 variants where available.

**Codebase Evidence:**

The `HammingDistanceFunctionFactory` is used for `int8` cell types, which in turn uses `vespalib::binary_hamming_distance`.

```cpp
// From: searchlib/src/vespa/searchlib/tensor/hamming_distance.cpp

template <typename VectorStoreType>
class BoundHammingDistance final : public BoundDistanceFunction {
    // ...
    double calc(TypedCells rhs) const noexcept override {
        // ...
        if constexpr (std::is_same<Int8Float, FloatType>::value) {
            return (double) vespalib::binary_hamming_distance(_lhs_vector.data(), rhs_vector.data(), sz);
        }
        // ...
    }
};
```

---

## 2. Analysis of Advanced Quantization Concepts

This section discusses advanced quantization techniques that are relevant in the field of vector search. The concepts were part of this research note, but the detailed C++ implementations described previously were illustrative examples, not representations of the actual Vespa codebase.

### 2.1. Product Quantization (PQ)

Product Quantization is a technique that divides a vector into several sub-vectors and quantizes each sub-vector independently using a small codebook. This allows for much higher compression ratios than scalar quantization. An Asymmetric Distance Computation (ADC) method can then be used to efficiently estimate distances between a query vector and the quantized database vectors.

**Implementation Status:**
> Codebase analysis confirms that Product Quantization (PQ), while a common and powerful technique, is **not implemented** in Vespa's core C++ `searchlib`. There are no native components for training PQ codebooks or performing ADC lookups. The detailed C++ class `ProductQuantizedAttribute` described in a previous version of this note was a hypothetical example.

### 2.2. Iterative Quantization (ITQ)

Iterative Quantization is a pre-processing step for binary quantization. It works by learning a rotation matrix that aligns the data with the axes before quantizing. This minimizes the quantization error and improves the performance of binary codes, especially for Euclidean or angular distance metrics. The process involves iteratively updating the binary codes and the rotation matrix until convergence.

**Implementation Status:**
> The `IterativeQuantization` class with Eigen matrix operations described previously was a theoretical example of how ITQ could be implemented. This technique is **not present** in the Vespa C++ codebase. Vespa's binary quantization support is focused on direct, optimized Hamming distance computation.

### 2.3. Other Advanced Concepts

Other advanced topics like **Mixed-Precision Quantization** (assigning different precisions to different parts of a vector based on importance) and **Streaming Quantization** (dynamically updating quantization parameters from a data stream) are active areas of research.

**Implementation Status:**
> These concepts are **not implemented** in the Vespa core C++ engine. The existing framework prioritizes robust, high-performance support for `int8` and binary types.