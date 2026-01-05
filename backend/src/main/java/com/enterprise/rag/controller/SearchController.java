package com.enterprise.rag.controller;


import java.util.Map;

import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;

@RestController
@RequestMapping("/api")
public class SearchController {
    
    private final RestTemplate restTemplate;
    private final String pythonServiceUrl = "http://localhost:8000";  // FastAPI
    
    public SearchController() {
        this.restTemplate = new RestTemplate();
    }
    
    @PostMapping("/search")
    public ResponseEntity<?> search(@RequestBody SearchRequest request) {
        // For now: userId = "demo-user"
        request.setUserId("demo-user");
        
        try {
            // Forward to Python RAG
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            
            HttpEntity<SearchRequest> entity = new HttpEntity<>(request, headers);
            ResponseEntity<String> response = restTemplate.postForEntity(
                pythonServiceUrl + "/chat",
                entity,
                String.class
            );
            
            return ResponseEntity.ok(response.getBody());
        } catch (Exception e) {
            return ResponseEntity.status(500).body(
                Map.of("error", "Python service unavailable: " + e.getMessage())
            );
        }
    }
    @PostMapping("/ingest")
public ResponseEntity<?> ingest(@RequestBody IngestRequest request) {
    request.setUserId("demo-user");
    
   try {
            // Forward to Python RAG
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);
            
            HttpEntity<IngestRequest> entity = new HttpEntity<>(request, headers);

            ResponseEntity<String> response = restTemplate.postForEntity(
                pythonServiceUrl + "/ingest",
                entity,
                String.class
            );
            
            return ResponseEntity.ok(response.getBody());
        } catch (Exception e) {
            return ResponseEntity.status(500).body(
                Map.of("error", "Python service unavailable: " + e.getMessage())
            );
        }
    }

}
