# Filterable HNSW 구현 분석: pgvector vs Qdrant

본 문서는 pgvector와 Qdrant의 filterable HNSW 구현 방식을 코드 레벨에서 분석하고 비교한 결과를 정리합니다.

## 1. 개요

Filterable HNSW는 벡터 유사도 검색과 조건부 필터링을 효율적으로 결합하는 기술입니다. 두 제품의 구현 방식을 분석한 결과, 근본적으로 다른 접근 방식을 취하고 있음을 발견했습니다.

## 2. Qdrant: 진정한 Filterable HNSW 구현

### 2.1 핵심 아키텍처

Qdrant는 전용 벡터 데이터베이스답게 filterable HNSW를 완전히 구현하고 있습니다.

#### FilteredScorer 구조체
```rust
pub struct FilteredScorer<'a> {
    raw_scorer: Box<dyn RawScorer + 'a>,
    filter_context: Option<BoxCow<'a, dyn FilterContext + 'a>>,
    point_deleted: &'a BitSlice,
    vec_deleted: &'a BitSlice,
    scores_buffer: Vec<ScoreType>,
}
```

#### 핵심 필터링 로직
```rust
pub fn check_vector(&self, point_id: PointOffsetType) -> bool {
    check_deleted_condition(point_id, self.vec_deleted, self.point_deleted)
        && self.filter_context.as_ref().is_none_or(|f| f.check(point_id))
}
```

### 2.2 동적 전략 선택 메커니즘

Qdrant는 **Cardinality Estimation**을 통해 필터 조건의 선택성을 분석하고 최적 전략을 동적으로 선택합니다.

```rust
// hnsw.rs의 search 로직에서
let query_cardinality = payload_index.estimate_cardinality(query_filter, &hw_counter);

if query_cardinality.max < self.config.full_scan_threshold {
    // 필터 결과가 적음 -> Plain Index 사용
    return self.search_vectors_plain(vectors, query_filter, top, params, query_context);
}

if query_cardinality.min > self.config.full_scan_threshold {
    // 필터 결과가 많음 -> HNSW + Filtering 사용  
    return self.search_vectors_with_graph(vectors, filter, top, params, query_context);
}

// 중간 영역 -> 샘플링을 통한 정확한 추정
```

### 2.3 통합 검색 과정

1. **필터 컨텍스트 생성**: 검색 전 필터 조건을 컨텍스트로 변환
2. **HNSW 탐색**: 각 노드 방문 시 필터 조건 동시 검사
3. **Early Termination**: 필터 조건 불만족 시 벡터 계산 생략
4. **결과 반환**: 이미 필터링된 결과만 반환

## 3. pgvector: 후처리 방식 필터링

### 3.1 PostgreSQL 통합 아키텍처

pgvector는 PostgreSQL 확장으로서 기존 인덱스 스캐닝 메커니즘을 활용합니다.

#### 인덱스 스캔 구조
```c
// hnswscan.c
IndexScanDesc hnswbeginscan(Relation index, int nkeys, int norderbys)
{
    // PostgreSQL 표준 인덱스 스캔 구조 사용
    scan = RelationGetIndexScan(index, nkeys, norderbys);
    // ...
}
```

#### 비트맵 스캔 미지원
```c
// hnsw.c, ivfflat.c
amroutine->amgetbitmap = NULL;  // 비트맵 스캔 비활성화
```

### 3.2 분리된 처리 방식

pgvector의 처리 순서:

1. **벡터 인덱스 스캔**: HNSW를 통한 유사도 기반 후보 검색
2. **PostgreSQL 엔진**: WHERE 절 조건을 별도로 적용  
3. **결과 조합**: 두 조건을 모두 만족하는 결과만 반환

```c
// GetScanItems에서 HNSW 검색 수행
static List *
GetScanItems(IndexScanDesc scan, Datum value)
{
    // HNSW 알고리즘으로 유사 벡터 검색
    return HnswSearchLayer(base, q, ep, hnsw_ef_search, 0, 
                          index, support, m, false, NULL, 
                          &so->v, hnsw_iterative_scan != HNSW_ITERATIVE_SCAN_OFF ? &so->discarded : NULL, 
                          true, &so->tuples);
}
```

### 3.3 한계점

- **Over-computation**: 필터링될 벡터도 유사도 계산 수행
- **고정 전략**: 데이터 분포와 무관한 일률적 처리
- **SQL 오버헤드**: PostgreSQL 쿼리 플래너의 추가 처리 비용

## 4. 상세 비교 분석

### 4.1 구현 방식 비교

| 구분 | Qdrant | pgvector |
|------|---------|----------|
| **아키텍처** | 전용 FilteredScorer | PostgreSQL IndexScan |
| **필터 통합** | HNSW 탐색 과정에 통합 | 벡터 검색 후 별도 적용 |
| **전략 선택** | 동적 (cardinality 기반) | 고정 (항상 동일) |
| **Early Termination** | 지원 | 미지원 |
| **메모리 효율성** | 높음 (불필요 계산 회피) | 낮음 (전체 계산 후 필터) |

### 4.2 성능 특성 분석

#### 높은 선택성 필터 (결과 < 1%)
- **Qdrant**: 
  - ✅ 매우 효율적 (99% 계산량 절약)
  - ✅ Plain search로 자동 전환 가능
  - ✅ 메모리 사용량 최소화

- **pgvector**: 
  - ❌ 비효율적 (99% 불필요 계산)
  - ❌ 고정된 HNSW 검색 수행
  - ❌ 높은 메모리 사용량

#### 낮은 선택성 필터 (결과 > 50%)
- **Qdrant**: 
  - ✅ 자동으로 HNSW + filtering 전환
  - ✅ 여전히 통합 처리로 효율성 유지

- **pgvector**: 
  - ⚠️ 상대적으로 합리적 성능
  - ❌ 여전히 분리된 처리로 인한 오버헤드

### 4.3 코드 복잡도 비교

#### Qdrant
```rust
// 복잡하지만 체계적인 구조
impl<'a> FilteredScorer<'a> {
    pub fn score_points(&mut self, point_ids: &mut Vec<PointOffsetType>, limit: usize) 
        -> impl Iterator<Item = ScoredPointOffset> 
    {
        // 필터링과 스코어링이 통합된 효율적 처리
        point_ids.retain(|point_id| self.check_vector(*point_id));
        if limit != 0 { point_ids.truncate(limit); }
        
        self.raw_scorer.score_points(point_ids, &mut self.scores_buffer[..point_ids.len()]);
        // ...
    }
}
```

#### pgvector
```c
// 단순하지만 분리된 구조
static List *
GetScanItems(IndexScanDesc scan, Datum value)
{
    // 단순한 HNSW 검색 (필터링 없음)
    return HnswSearchLayer(base, q, ep, hnsw_ef_search, 0, index, support, m, 
                          false, NULL, &so->v, &so->discarded, true, &so->tuples);
}
// WHERE 절은 PostgreSQL 엔진이 별도 처리
```

## 5. 실제 사용 사례별 권장사항

### 5.1 Qdrant 권장 시나리오
- ✅ **높은 선택성 필터**: 검색 결과의 10% 미만을 필터링하는 경우
- ✅ **복잡한 필터 조건**: 다중 조건, 범위 검색, 지리적 필터 등
- ✅ **대용량 데이터**: 수백만 개 이상의 벡터에서 정밀한 필터링
- ✅ **성능 최우선**: 응답 시간과 처리량이 핵심인 서비스

### 5.2 pgvector 권장 시나리오  
- ✅ **PostgreSQL 생태계**: 기존 PostgreSQL 인프라 활용 필수
- ✅ **복합 쿼리**: 벡터 검색과 복잡한 SQL 쿼리의 조합
- ✅ **트랜잭션 요구**: ACID 속성이 중요한 애플리케이션
- ✅ **단순한 필터**: 기본적인 WHERE 절 조건

## 6. 기술적 통찰

### 6.1 설계 철학의 차이
- **Qdrant**: "벡터 검색 최적화 우선, 필터링 완전 통합"
- **pgvector**: "SQL 생태계 호환성 우선, 기존 메커니즘 활용"

### 6.2 성능 vs 호환성 트레이드오프
- **Qdrant**: 높은 성능, 전용 API 학습 필요
- **pgvector**: 표준 SQL 사용, 성능 제약 존재

### 6.3 미래 발전 방향
- **Qdrant**: 더 정교한 cardinality estimation, GPU 가속 필터링
- **pgvector**: PostgreSQL 16+의 새로운 인덱스 기능 활용 가능성

## 7. 결론

두 제품의 filterable HNSW 구현은 근본적으로 다른 접근 방식을 보여줍니다:

- **Qdrant**는 현대적인 벡터 검색 연구 성과를 반영한 **진정한 filterable HNSW**를 구현하여, 필터 조건이 엄격할수록 뛰어난 성능을 보입니다.

- **pgvector**는 PostgreSQL 생태계와의 완벽한 호환성을 위해 **후처리 방식**을 채택하여, SQL의 강력함을 활용할 수 있지만 순수 벡터 검색 성능에서는 제약이 있습니다.

선택 기준은 **성능 최적화 vs SQL 생태계 활용** 중 어느 것이 더 중요한지에 따라 결정되어야 합니다.

---

*본 분석은 2025년 1월 기준 pgvector와 Qdrant의 소스코드를 직접 분석한 결과입니다.* 