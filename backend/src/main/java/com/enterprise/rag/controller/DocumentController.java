package com.enterprise.rag.controller;
import java.util.List;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.enterprise.rag.model.Document;
import com.enterprise.rag.model.User;
import com.enterprise.rag.repository.DocumentRepository;
import com.enterprise.rag.repository.UserRepository;

@RestController
@RequestMapping("/api")
public class DocumentController {
    
    private final DocumentRepository docRepo;
    private final UserRepository userRepo;
    
    public DocumentController(DocumentRepository docRepo, UserRepository userRepo) {
        this.docRepo = docRepo;
        this.userRepo = userRepo;
    }
    
    @GetMapping("/users")
    public List<User> allUsers() {
        return userRepo.findAll();
    }
    
    @GetMapping("/documents")
    public List<Document> allDocuments() {
        return docRepo.findAll();
    }
    
    @GetMapping("/user/{name}/documents")
    public ResponseEntity<List<Document>> userDocuments(@PathVariable String name) {
    User user = userRepo.findByName(name);
    if (user == null) {
        return ResponseEntity.notFound().build();
    }
    
    // ✅ Join with user_document_access
    List<Document> userDocs = docRepo.findByUserId(user.getId());
    return ResponseEntity.ok(userDocs);
}
}
