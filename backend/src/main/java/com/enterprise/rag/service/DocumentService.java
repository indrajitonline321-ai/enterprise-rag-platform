package com.enterprise.rag.service;

import java.util.Map;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import com.azure.storage.blob.BlobClient;
import com.azure.storage.blob.BlobContainerClient;
import com.azure.storage.blob.BlobServiceClient;
import com.azure.storage.blob.BlobServiceClientBuilder;
import com.enterprise.rag.model.Document;
import com.enterprise.rag.model.User;
import com.enterprise.rag.model.UserDocumentAccess;
import com.enterprise.rag.repository.DocumentRepository;
import com.enterprise.rag.repository.UserDocumentAccessRepository;
import com.enterprise.rag.repository.UserRepository;


@Service
public class DocumentService {
    
    @Value("${azure.storage.connection-string}")
    private String connectionString;

    private final DocumentRepository docRepo;
    private final UserRepository userRepo;
    private final UserDocumentAccessRepository accessRepo;

    // ✅ Simple constructor - only repos needed
    public DocumentService(DocumentRepository docRepo,
                          UserRepository userRepo,
                          UserDocumentAccessRepository accessRepo) {
        this.docRepo = docRepo;
        this.userRepo = userRepo;
        this.accessRepo = accessRepo;
    }

    // ✅ uploadToAzure stays the same (uses @Value connectionString)
    private String uploadToAzure(MultipartFile file, String blobName) {
        try {
            BlobServiceClient service = new BlobServiceClientBuilder()
                .connectionString(connectionString)
                .buildClient();
                
            BlobContainerClient container = service.getBlobContainerClient("documents");
            container.createIfNotExists();
            
            BlobClient blob = container.getBlobClient(blobName);
            blob.upload(file.getInputStream(), file.getSize(), true);
            
            return blob.getBlobUrl();
        } catch (Exception e) {
            throw new RuntimeException("Failed to upload to Azure Blob Storage", e);
        }
    }

    public Map<String, String> uploadAndGrant(MultipartFile file, String uploaderName) {
        String blobName = file.getOriginalFilename();
        String blobUrl = uploadToAzure(file, blobName);

        Document doc = new Document();
        doc.setName(file.getOriginalFilename());
        doc.setBlobName(blobName);
        doc.setBlobUrl(blobUrl);
        doc = docRepo.save(doc);

        User user = userRepo.findByName(uploaderName);
        if (user == null) {
            user = new User();
            user.setName(uploaderName);
            user.setRole("user");
            user = userRepo.save(user);
        }

        UserDocumentAccess access = new UserDocumentAccess();
        access.setUser(user);
        access.setDocument(doc);
        accessRepo.save(access);

        return Map.of("documentId", String.valueOf(doc.getId()),
                "blobUrl", blobUrl
        );
    }
}
