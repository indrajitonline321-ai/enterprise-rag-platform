package com.enterprise.rag.controller;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonProperty;

import lombok.Data;

@Data
public class SearchRequest {
    private String query;
    private String userId;
    @JsonProperty("limit")
    private Integer limit = 5;

    private List<String> docIds;
    
}