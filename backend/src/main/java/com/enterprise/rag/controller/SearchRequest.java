package com.enterprise.rag.controller;

import com.fasterxml.jackson.annotation.JsonProperty;

import lombok.Data;

@Data
public class SearchRequest {
    private String query;
    private String userId;
    @JsonProperty("limit")
    private Integer limit = 5;
}