package com.enterprise.rag.controller;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping; 
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import com.azure.storage.blob.BlobContainerClient;
import com.azure.storage.blob.BlobServiceClient;
import com.azure.storage.blob.BlobServiceClientBuilder;
import com.azure.storage.blob.models.BlobItem;
import com.enterprise.rag.service.DocumentService;

@RestController
@RequestMapping("/api/files")
public class FileController {

    @Value("${azure.storage.connection-string}")
    private String connectionString;

    private final DocumentService documentService;  // ✅ add field

    public FileController(DocumentService documentService) {  // ✅ constructor injection
        this.documentService = documentService;
    }

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
public ResponseEntity<Map<String, String>> upload(
        @RequestParam("file") MultipartFile file,
        @RequestParam(defaultValue = "Admin") String userName) {
    return ResponseEntity.ok(documentService.uploadAndGrant(file, userName));
}
}


