package com.enterprise.rag.controller;

import com.azure.storage.blob.*;
import com.azure.storage.blob.models.BlobItem;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile; 

import java.util.ArrayList;
import java.util.List;
import java.util.*;

@RestController
@RequestMapping("/api/files")
public class FileController {

    @Value("${azure.storage.connection-string}")
    private String connectionString;

    @GetMapping("/list")
    public ResponseEntity<List<String>> list() {
        BlobServiceClient service = new BlobServiceClientBuilder()
                .connectionString(connectionString)
                .buildClient();

        BlobContainerClient container = service.getBlobContainerClient("documents");
        if (!container.exists()) {
            return ResponseEntity.ok(List.of());
        }

        List<String> names = new ArrayList<>();
        for (BlobItem item : container.listBlobs()) {
            names.add(item.getName());
        }
        return ResponseEntity.ok(names);
    }
    @PostMapping("/upload")  
    public ResponseEntity<Map<String, String>> upload(@RequestParam("file") MultipartFile file) {
        try {
            BlobServiceClient service = new BlobServiceClientBuilder()
                .connectionString(connectionString)
                .buildClient();
                
            BlobContainerClient container = service.getBlobContainerClient("documents");
            container.createIfNotExists();
            
            BlobClient blob = container.getBlobClient(file.getOriginalFilename());
            blob.upload(file.getInputStream(), file.getSize(), true);
            
            Map<String, String> response = Map.of(
                "message", "✅ File uploaded successfully!",
                "name", file.getOriginalFilename(),
                "url", blob.getBlobUrl()
            );
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            return ResponseEntity.internalServerError()
                .body(Map.of("error", e.getMessage()));
        }
}
}

