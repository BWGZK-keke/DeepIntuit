include "base.thrift"
namespace go tns.data_infra.feature_service

typedef string JsonDict

struct FeatureRequest {
  1: required string id,
  2: required string scene,

  255: optional base.Base Base
}

struct RawFeatureResponse {
  1: required JsonDict rawFeature,

  255: optional base.BaseResp BaseResp
}

service FeatureFetchService {
  RawFeatureResponse FetchRawFeature(1: FeatureRequest req)
}
