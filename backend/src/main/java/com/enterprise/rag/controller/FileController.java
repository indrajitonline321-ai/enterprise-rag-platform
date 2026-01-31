package com.enterprise.rag.controller;

import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping; 
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import com.azure.storage.blob.BlobClient;
import com.azure.storage.blob.BlobContainerClient;
import com.azure.storage.blob.BlobServiceClient;
import com.azure.storage.blob.BlobServiceClientBuilder;
import com.azure.storage.blob.models.BlobItem;
import com.enterprise.rag.model.ContactRequest;
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

    @GetMapping("/download/{blobName}")
public ResponseEntity<byte[]> downloadFile(@PathVariable String blobName) {
    byte[] data = downloadFromAzure(blobName);

    HttpHeaders headers = new HttpHeaders();
    headers.setContentType(MediaType.APPLICATION_OCTET_STREAM);
    headers.setContentLength(data.length);
    headers.setContentDispositionFormData("attachment", blobName);

    return new ResponseEntity<>(data, headers, HttpStatus.OK);
}

private byte[] downloadFromAzure(String blobName) {
    try {
        BlobServiceClient service = new BlobServiceClientBuilder()
            .connectionString(connectionString)
            .buildClient();

        BlobContainerClient container = service.getBlobContainerClient("documents");
        BlobClient blob = container.getBlobClient(blobName);

        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        blob.downloadStream(outputStream);   // SDK v12 method
        return outputStream.toByteArray();
    } catch (Exception e) {
        throw new RuntimeException("Failed to download from Azure Blob Storage", e);
    }
}

    

@PostMapping("/upload")  
public ResponseEntity<Map<String, String>> upload(
        @RequestParam("file") MultipartFile file,
        @RequestParam(defaultValue = "Admin") String userName) {
    return ResponseEntity.ok(documentService.uploadAndGrant(file, userName));
}

@PostMapping("/contactUS")  
public ResponseEntity<String> contactUS(@RequestBody ContactRequest request) {
    return ResponseEntity.ok(documentService.saveUserQuery(
        request.getName(), 
        request.getEmail(), 
        request.getQuery()
    ));
}

}


