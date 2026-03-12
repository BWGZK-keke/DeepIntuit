#!/usr/local/bin/thrift --gen py
include "base.thrift"

typedef string JsonDict

struct ProcessRequest {
    1: string config_key
    2: required list<string> pos_ids,
    255: optional base.Base Base,
}

struct ProcessResponse {
    1: JsonDict regex_suggestion,
    2: optional map<string, JsonDict>  extra,
    255: optional base.BaseResp BaseResp
}

service TiktokAuraRegexService {
  ProcessResponse process(1: ProcessRequest req)
}
