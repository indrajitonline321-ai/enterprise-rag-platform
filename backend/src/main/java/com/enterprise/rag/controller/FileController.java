package com.enterprise.rag.controller;

import com.azure.storage.blob.*;
import com.azure.storage.blob.models.BlobItem;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.List;

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
}

