# Project Voyager: Tech Roadmap Considerations

**Last updated:** Jan 5, 2026  
**Purpose:** Core Agentic Technical Gaps in Voyager  
**Author(s):** Popin Bose Roy  
**Reviewer(s):** Austin Byrne, Herman Chiang, Shivani Rawat, Solon Mao, Teresa Thomas, Julia Kanter

---

## Background

To deliver the Voyager H1 Experience Roadmap, we must evolve our mental model of Voyager—from a conversational loop to an **Operating System**.

| OS Analogy | Voyager Component |
|------------|-------------------|
| **Operating System** | The Core Voyager Platform—orchestrating intent, memory, and sub-agents |
| **Applications** | H1 experiences: Search, Auto-suggest, Intent Mapping, Dead-end Recovery |
| **Hard Drive** | Persistent Memory—long-term storage of user preferences and conversation history |
| **RAM** | Context Window—working memory that powers each conversational turn |

### The Challenge

Voyager is not currently architected to support multiple concurrent complex sub-agents. As we ship new experiences, each component (Property Search, HDP modules, Omnibar follow-ups) will compete for the same finite context window. Research shows that long-context degrades LLM performance—without active management, we risk optimizing for local maxima while overall user experience suffers.

Our deep investments in Memory Orchestration will set us up for long-term success. However, in the near and mid-term, we must address five critical technical gaps:

1. **Context Window Utilization Management** — No systematic arbitration for allocating finite context across competing sub-agents
2. **Representation of Voyager AI Responses** — Responses often swing between over-verbose and over-concise, impacting multi-turn engagement
3. **Scalable & Modular AI Architecture** — Third-party AI infrastructure constraints and lack of LLM-agnostic design
4. **AI Behavior Learning & Measurement** — No shared framework to capture and socialize conversational behavior patterns across teams
5. **System Degradation Strategy** — Missing unified approach to handle failures gracefully and maintain user trust

### Why a Coordinated Strategy

These gaps are interconnected. Context mismanagement affects response quality. Poor response representation drives single-turn abandonment. Without shared behavioral metrics, teams duplicate experiments. Without degradation strategies, failures cascade unpredictably.

A coordinated technical strategy—not disparate fixes—will transform the AI layer from a potential bottleneck into Zillow's competitive advantage. As we scale to millions of users, these investments ensure we deliver the right context and information every time, doing the right thing for Beths and Rachels.

---

## Alternative Version (More Concise)

If you prefer a tighter version:

---

## Background

To deliver the Voyager H1 Experience Roadmap, we must evolve how we think about Voyager—from a conversational loop to an **Operating System**:

- **OS**: The Core Voyager Platform  
- **Apps**: H1 experiences (Search, Auto-suggest, Intent Mapping)  
- **Hard Drive**: Persistent Memory (long-term user context)  
- **RAM**: Context Window (working memory per turn)

Today, Voyager is not architected to support multiple concurrent sub-agents. Each experience—Property Search, HDP modules, Omnibar—competes for a finite context window. Without systematic management, we optimize for local maxima while overall experience degrades.

Our Memory Orchestration investments address long-term needs, but near-term success requires closing five critical gaps:

| Gap | Risk if Unaddressed |
|-----|---------------------|
| **Context Window Management** | Sub-agents compete for tokens without arbitration; user intent gets starved |
| **Response Representation** | Verbosity swings block multi-turn conversations (>40% queries are single-turn) |
| **Scalable AI Architecture** | Vendor lock-in prevents rapid LLM adoption; no unified fallback mechanisms |
| **Behavior Learning & Measurement** | Teams duplicate experiments; conversational patterns go unobserved |
| **System Degradation Strategy** | Silent failures erode trust; no defined fallback ladder |

These gaps are interconnected—context mismanagement degrades response quality, poor responses drive abandonment, and without shared metrics, we learn slowly. A coordinated technical strategy will transform the AI layer from bottleneck to competitive advantage, ensuring we deliver the right context every time as we scale to millions of users.

---
