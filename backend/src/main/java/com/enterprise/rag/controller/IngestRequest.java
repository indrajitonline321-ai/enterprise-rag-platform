package com.enterprise.rag.controller;

import lombok.Data;
@Data  // Lombok
public class IngestRequest {
    private String document_id;
    private String blob_url;
    private String user_id;
}
