# Response Latency Perception Study - Analysis Plan
# Complete R script for analyzing Qualtrics data

library(tidyverse)
library(lme4)
library(lmerTest)
library(ggplot2)
library(broom.mixed)
library(emmeans)
library(mediation)

# Load and prepare data
load_qualtrics_data <- function(file_path) {
  # Load raw Qualtrics data
  raw_data <- read.csv(file_path, stringsAsFactors = FALSE)
  
  # Remove first two rows (Qualtrics headers)
  data <- raw_data[-(1:2), ]
  
  # Convert to long format for trial-level analysis
  trial_data <- data %>%
    select(ResponseId, starts_with("trial_"), starts_with("quality_rating_"), 
           starts_with("wait_felt_"), starts_with("wait_hurt_quality_"), 
           starts_with("would_accept_"), starts_with("perceived_wait_"),
           max_acceptable_wait, streaming_preference) %>%
    pivot_longer(
      cols = c(starts_with("quality_rating_"), starts_with("wait_felt_"), 
               starts_with("wait_hurt_quality_"), starts_with("would_accept_")),
      names_to = c("measure", "trial"),
      names_pattern = "(.+)_(\\d+)",
      values_to = "rating"
    ) %>%
    pivot_wider(names_from = measure, values_from = rating) %>%
    mutate(
      trial = as.numeric(trial),
      quality_rating = as.numeric(quality_rating),
      wait_felt = as.numeric(wait_felt),
      wait_hurt_quality = as.numeric(wait_hurt_quality),
      would_accept = as.numeric(would_accept),
      accept_binary = ifelse(would_accept >= 5, 1, 0)
    )
  
  # Add condition information
  for (i in 1:5) {
    trial_data <- trial_data %>%
      mutate(
        prompt_id = ifelse(trial == i, data[[paste0("trial_", i, "_prompt_id")]], prompt_id),
        modality = ifelse(trial == i, data[[paste0("trial_", i, "_modality")]], modality),
        latency_ms = ifelse(trial == i, as.numeric(data[[paste0("trial_", i, "_latency_ms")]]), latency_ms),
        actual_latency_ms = ifelse(trial == i, as.numeric(data[[paste0("trial_", i, "_actual_latency_ms")]]), actual_latency_ms)
      )
  }
  
  trial_data <- trial_data %>%
    mutate(
      latency_sec = latency_ms / 1000,
      actual_latency_sec = actual_latency_ms / 1000,
      latency_level = case_when(
        latency_ms == 500 ~ "Fast",
        latency_ms == 2000 ~ "Medium", 
        latency_ms == 6000 ~ "Slow"
      ),
      latency_level = factor(latency_level, levels = c("Fast", "Medium", "Slow")),
      modality = factor(modality, levels = c("nonstream", "stream")),
      prompt_id = factor(prompt_id)
    )
  
  return(trial_data)
}

# 1. DESCRIPTIVE STATISTICS
descriptive_stats <- function(data) {
  # Overall means by condition
  condition_means <- data %>%
    group_by(latency_level, modality) %>%
    summarise(
      n_obs = n(),
      quality_mean = mean(quality_rating, na.rm = TRUE),
      quality_sd = sd(quality_rating, na.rm = TRUE),
      wait_felt_mean = mean(wait_felt, na.rm = TRUE),
      wait_felt_sd = sd(wait_felt, na.rm = TRUE),
      acceptance_rate = mean(accept_binary, na.rm = TRUE),
      .groups = 'drop'
    )
  
  print("Condition Means:")
  print(condition_means)
  
  # Participant-level summaries
  participant_summary <- data %>%
    group_by(ResponseId) %>%
    summarise(
      max_acceptable = first(max_acceptable_wait),
      streaming_pref = first(streaming_preference),
      .groups = 'drop'
    )
  
  print("Max Acceptable Wait (seconds):")
  print(summary(as.numeric(participant_summary$max_acceptable)))
  
  print("Streaming Preference:")
  print(table(participant_summary$streaming_pref))
  
  return(list(condition_means = condition_means, participant_summary = participant_summary))
}

# 2. MAIN EFFECTS ANALYSIS
analyze_main_effects <- function(data) {
  # Quality rating analysis
  quality_model <- lmer(quality_rating ~ latency_level * modality + 
                       (1 | ResponseId) + (1 | prompt_id), 
                       data = data)
  
  print("QUALITY RATING MODEL:")
  print(summary(quality_model))
  print(anova(quality_model))
  
  # Wait perception analysis  
  wait_model <- lmer(wait_felt ~ latency_level * modality + 
                    (1 | ResponseId) + (1 | prompt_id), 
                    data = data)
  
  print("WAIT PERCEPTION MODEL:")
  print(summary(wait_model))
  print(anova(wait_model))
  
  # Acceptance analysis (logistic)
  accept_model <- glmer(accept_binary ~ latency_level * modality + 
                       (1 | ResponseId) + (1 | prompt_id), 
                       family = binomial, data = data)
  
  print("ACCEPTANCE MODEL:")
  print(summary(accept_model))
  print(anova(accept_model))
  
  return(list(quality = quality_model, wait = wait_model, accept = accept_model))
}

# 3. LATENCY THRESHOLD ANALYSIS (L50)
calculate_l50 <- function(data) {
  # Fit logistic models for each modality
  nonstream_data <- filter(data, modality == "nonstream")
  stream_data <- filter(data, modality == "stream")
  
  # Non-streaming L50
  nonstream_model <- glmer(accept_binary ~ latency_sec + (1 | ResponseId) + (1 | prompt_id), 
                          family = binomial, data = nonstream_data)
  
  nonstream_l50 <- -coef(summary(nonstream_model))[1,1] / coef(summary(nonstream_model))[2,1]
  
  # Streaming L50 (only medium and slow)
  stream_model <- glmer(accept_binary ~ latency_sec + (1 | ResponseId) + (1 | prompt_id), 
                       family = binomial, data = stream_data)
  
  stream_l50 <- -coef(summary(stream_model))[1,1] / coef(summary(stream_model))[2,1]
  
  l50_benefit <- stream_l50 - nonstream_l50
  
  print(paste("Non-streaming L50:", round(nonstream_l50, 2), "seconds"))
  print(paste("Streaming L50:", round(stream_l50, 2), "seconds"))
  print(paste("Streaming benefit:", round(l50_benefit, 2), "seconds"))
  
  return(list(nonstream_l50 = nonstream_l50, stream_l50 = stream_l50, benefit = l50_benefit))
}

# 4. PERCEIVED VS ACTUAL LATENCY
analyze_perceived_latency <- function(data) {
  # Add perceived latency data (this would need to be merged from slider responses)
  # For now, simulate the analysis structure
  
  perceived_data <- data %>%
    mutate(
      # This would be actual perceived latency from sliders
      perceived_latency_bias = NA, # perceived_sec - actual_latency_sec
      streaming_reduces_bias = modality == "stream"
    )
  
  # Model perceived latency bias
  bias_model <- lmer(perceived_latency_bias ~ latency_level * modality + 
                    (1 | ResponseId) + (1 | prompt_id), 
                    data = perceived_data)
  
  print("PERCEIVED LATENCY BIAS MODEL:")
  print("(Note: This requires merging slider data)")
  # print(summary(bias_model))
  
  return(bias_model)
}

# 5. MEDIATION ANALYSIS
mediation_analysis <- function(data) {
  # Test if perceived wait mediates latency -> quality relationship
  
  # Step 1: Latency -> Quality (total effect)
  total_model <- lmer(quality_rating ~ latency_sec + (1 | ResponseId) + (1 | prompt_id), 
                     data = data)
  
  # Step 2: Latency -> Perceived Wait (a path)
  a_model <- lmer(wait_felt ~ latency_sec + (1 | ResponseId) + (1 | prompt_id), 
                 data = data)
  
  # Step 3: Latency + Perceived Wait -> Quality (b path and direct effect)
  mediation_model <- lmer(quality_rating ~ latency_sec + wait_felt + 
                         (1 | ResponseId) + (1 | prompt_id), 
                         data = data)
  
  print("MEDIATION ANALYSIS:")
  print("Total effect (c):")
  print(summary(total_model))
  print("a path (latency -> wait perception):")
  print(summary(a_model))
  print("Mediation model (c' and b paths):")
  print(summary(mediation_model))
  
  return(list(total = total_model, a_path = a_model, mediation = mediation_model))
}

# 6. VISUALIZATION FUNCTIONS
create_visualizations <- function(data, models) {
  # Quality by condition
  p1 <- ggplot(data, aes(x = latency_level, y = quality_rating, fill = modality)) +
    geom_boxplot(alpha = 0.7) +
    geom_point(position = position_jitterdodge(dodge.width = 0.75), alpha = 0.3) +
    labs(title = "Response Quality by Latency and Modality",
         x = "Latency Level", y = "Quality Rating (1-7)",
         fill = "Response Type") +
    theme_minimal() +
    scale_fill_brewer(palette = "Set2", labels = c("Non-streaming", "Streaming"))
  
  # Acceptance rates
  accept_summary <- data %>%
    group_by(latency_level, modality) %>%
    summarise(acceptance_rate = mean(accept_binary, na.rm = TRUE), .groups = 'drop')
  
  p2 <- ggplot(accept_summary, aes(x = latency_level, y = acceptance_rate, 
                                  color = modality, group = modality)) +
    geom_line(size = 1.2) +
    geom_point(size = 3) +
    labs(title = "Acceptance Rates by Condition",
         x = "Latency Level", y = "Acceptance Rate",
         color = "Response Type") +
    theme_minimal() +
    scale_color_brewer(palette = "Set1", labels = c("Non-streaming", "Streaming")) +
    scale_y_continuous(labels = scales::percent)
  
  # Wait perception
  p3 <- ggplot(data, aes(x = latency_sec, y = wait_felt, color = modality)) +
    geom_point(alpha = 0.4) +
    geom_smooth(method = "lm", se = TRUE) +
    labs(title = "Perceived Wait Time vs Actual Latency",
         x = "Actual Latency (seconds)", y = "Perceived Wait (1-7 scale)",
         color = "Response Type") +
    theme_minimal() +
    scale_color_brewer(palette = "Set1", labels = c("Non-streaming", "Streaming"))
  
  return(list(quality_plot = p1, acceptance_plot = p2, perception_plot = p3))
}

# 7. MAIN ANALYSIS FUNCTION
run_complete_analysis <- function(file_path) {
  # Load data
  cat("Loading and preparing data...\n")
  data <- load_qualtrics_data(file_path)
  
  # Descriptive statistics
  cat("\n=== DESCRIPTIVE STATISTICS ===\n")
  descriptives <- descriptive_stats(data)
  
  # Main effects
  cat("\n=== MAIN EFFECTS ANALYSIS ===\n")
  models <- analyze_main_effects(data)
  
  # L50 analysis
  cat("\n=== LATENCY THRESHOLD (L50) ANALYSIS ===\n")
  l50_results <- calculate_l50(data)
  
  # Perceived latency
  cat("\n=== PERCEIVED LATENCY ANALYSIS ===\n")
  perceived_model <- analyze_perceived_latency(data)
  
  # Mediation
  cat("\n=== MEDIATION ANALYSIS ===\n")
  mediation_results <- mediation_analysis(data)
  
  # Visualizations
  cat("\n=== CREATING VISUALIZATIONS ===\n")
  plots <- create_visualizations(data, models)
  
  # Print plots
  print(plots$quality_plot)
  print(plots$acceptance_plot)
  print(plots$perception_plot)
  
  # Return all results
  return(list(
    data = data,
    descriptives = descriptives,
    models = models,
    l50 = l50_results,
    mediation = mediation_results,
    plots = plots
  ))
}

# EXAMPLE USAGE:
# results <- run_complete_analysis("qualtrics_data.csv")

# EFFECT SIZE CALCULATIONS
calculate_effect_sizes <- function(models) {
  # Cohen's d for quality differences
  quality_emmeans <- emmeans(models$quality, ~ latency_level | modality)
  quality_contrasts <- contrast(quality_emmeans, "pairwise")
  
  print("EFFECT SIZES - Quality Rating Differences:")
  print(quality_contrasts)
  
  # Odds ratios for acceptance
  accept_emmeans <- emmeans(models$accept, ~ latency_level | modality, type = "response")
  accept_contrasts <- contrast(accept_emmeans, "pairwise")
  
  print("EFFECT SIZES - Acceptance Odds Ratios:")
  print(accept_contrasts)
}

# POWER ANALYSIS (POST-HOC)
power_analysis <- function(data) {
  # Calculate observed effect sizes and power
  library(pwr)
  
  # Example for quality rating differences
  quality_summary <- data %>%
    group_by(latency_level, modality) %>%
    summarise(
      mean_quality = mean(quality_rating, na.rm = TRUE),
      sd_quality = sd(quality_rating, na.rm = TRUE),
      n = n(),
      .groups = 'drop'
    )
  
  print("POWER ANALYSIS - Observed Effect Sizes:")
  print(quality_summary)
  
  # Calculate Cohen's d between conditions
  fast_vs_slow <- quality_summary %>% filter(latency_level %in% c("Fast", "Slow"))
  if(nrow(fast_vs_slow) >= 2) {
    pooled_sd <- sqrt(mean(fast_vs_slow$sd_quality^2))
    cohens_d <- diff(fast_vs_slow$mean_quality) / pooled_sd
    
    print(paste("Cohen's d (Fast vs Slow):", round(cohens_d, 3)))
    
    # Post-hoc power
    power_result <- pwr.t.test(n = 100, d = abs(cohens_d), sig.level = 0.05)
    print(paste("Achieved power:", round(power_result$power, 3)))
  }
}

cat("Analysis script loaded. Run: results <- run_complete_analysis('your_data.csv')\n")