package com.enterprise.rag.model;

import lombok.Data;

@Data  // Lombok (add if using)
public class ContactRequest {
    private String name;
    private String email;
    private String query;
}
