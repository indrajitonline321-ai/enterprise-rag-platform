package com.enterprise.rag;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.domain.EntityScan;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.data.jpa.repository.config.EnableJpaRepositories;

@SpringBootApplication
@ComponentScan(basePackages = "com.enterprise.rag")  // ✅ Scans service, repository, controller
@EntityScan("com.enterprise.rag.model")              // ✅ Scans entities
@EnableJpaRepositories("com.enterprise.rag.repository") 
public class EnterpriseRagBackendApplication {

	public static void main(String[] args) {
		SpringApplication.run(EnterpriseRagBackendApplication.class, args);
	}

}


