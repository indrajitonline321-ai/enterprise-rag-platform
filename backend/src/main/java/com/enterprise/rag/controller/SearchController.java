package com.enterprise.rag.controller;


import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;

import com.enterprise.rag.model.User;
import com.enterprise.rag.repository.DocumentRepository;
import com.enterprise.rag.repository.UserRepository;

@RestController
@RequestMapping("/api")
public class SearchController {
    
    private final RestTemplate restTemplate;
    @Value("${PYTHON_SERVICE_URL:http://localhost:8000}")
    private String pythonServiceUrl;   // FastAPI
    private final UserRepository userRepo;
    private final DocumentRepository docRepo;
    

      public SearchController(UserRepository userRepo, DocumentRepository docRepo) {
        this.userRepo = userRepo;
        this.docRepo = docRepo;
        this.restTemplate = new RestTemplate();

    }

        
    
    @PostMapping("/search")
    public ResponseEntity<?> search(@RequestBody SearchRequest request) {
        // For now: userId = "demo-user"

    User user = userRepo.findByName(request.getUserId());
    
    request.setUserId(String.valueOf(user.getId()));
  
    List<String> docIds = new ArrayList<>(); 
    docIds.add(String.valueOf(user.getId()));

// 3. Set the list into the request
    request.setDocIds(docIds);
    

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
    public ResponseEntity<String> ingest(@RequestBody IngestRequest request, 
                                   @RequestParam(defaultValue="demo-user") String userName) {
    
   try {
        System.err.println(request+"  userName    "     +userName);
          User user = userRepo.findByName(request.getUser_id());
            request.setUser_id(String.valueOf(user.getId()));

    // 2. Check blob_url belongs to user
  
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
   }
        
        catch (Exception e) {
            return ResponseEntity.status(500).body(e.getMessage());
        }
    }

}
