"""Tests for scoring engine."""

from src.domain.models import ChannelConfig, ScoringFeatures, SlackMessage
from src.services import scoring_engine


def test_extract_features_keywords(
    sample_slack_message: SlackMessage, sample_channel_config: ChannelConfig
) -> None:
    """Test feature extraction with keywords."""
    features = scoring_engine.extract_features(
        sample_slack_message, sample_channel_config
    )

    assert features.has_keywords is True
    assert features.keyword_count >= 1  # "release" is in text


def test_extract_features_reactions(
    sample_slack_message: SlackMessage, sample_channel_config: ChannelConfig
) -> None:
    """Test feature extraction counts reactions."""
    features = scoring_engine.extract_features(
        sample_slack_message, sample_channel_config
    )

    assert features.reaction_count == 7  # 5 + 2 from fixture


def test_extract_features_replies(
    sample_slack_message: SlackMessage, sample_channel_config: ChannelConfig
) -> None:
    """Test feature extraction counts replies."""
    features = scoring_engine.extract_features(
        sample_slack_message, sample_channel_config
    )

    assert features.reply_count == 3


def test_extract_features_anchors(
    sample_slack_message: SlackMessage, sample_channel_config: ChannelConfig
) -> None:
    """Test feature extraction counts anchors."""
    features = scoring_engine.extract_features(
        sample_slack_message, sample_channel_config
    )

    assert features.anchor_count == 1


def test_extract_features_sets_has_files_from_attachments(
    sample_slack_message: SlackMessage, sample_channel_config: ChannelConfig
) -> None:
    """Attachments should set the has_files flag."""
    message_with_attachment = sample_slack_message.model_copy(
        update={"attachments_count": 1}
    )

    features = scoring_engine.extract_features(
        message_with_attachment, sample_channel_config
    )

    assert features.has_files is True


def test_extract_features_sets_has_files_from_files(
    sample_slack_message: SlackMessage, sample_channel_config: ChannelConfig
) -> None:
    """Slack file uploads should set the has_files flag."""
    message_with_file = sample_slack_message.model_copy(update={"files_count": 2})

    features = scoring_engine.extract_features(
        message_with_file, sample_channel_config
    )

    assert features.has_files is True


def test_calculate_score_keywords(sample_channel_config: ChannelConfig) -> None:
    """Test score calculation with keywords."""
    features = ScoringFeatures(
        has_keywords=True,
        keyword_count=2,
        has_mention=False,
        reply_count=0,
        reaction_count=0,
        anchor_count=0,
        link_count=0,
        has_files=False,
        is_bot=False,
        channel_name="test",
    )

    score = scoring_engine.calculate_score(features, sample_channel_config)

    # 2 keywords * 10.0 = 20.0
    assert score == 20.0


def test_calculate_score_mention(sample_channel_config: ChannelConfig) -> None:
    """Test score calculation with channel mention."""
    features = ScoringFeatures(
        has_keywords=False,
        keyword_count=0,
        has_mention=True,
        reply_count=0,
        reaction_count=0,
        anchor_count=0,
        link_count=0,
        has_files=False,
        is_bot=False,
        channel_name="test",
    )

    score = scoring_engine.calculate_score(features, sample_channel_config)

    # Mention weight = 8.0
    assert score == 8.0


def test_calculate_score_bot_penalty(sample_channel_config: ChannelConfig) -> None:
    """Test bot penalty."""
    features = ScoringFeatures(
        has_keywords=True,
        keyword_count=1,
        has_mention=False,
        reply_count=0,
        reaction_count=0,
        anchor_count=0,
        link_count=0,
        has_files=False,
        is_bot=True,
        channel_name="test",
    )

    score = scoring_engine.calculate_score(features, sample_channel_config)

    # 10 (keyword) - 15 (bot penalty) = -5
    assert score == -5.0
    assert "bot penalty applied" in features.explanations


def test_calculate_score_attachments_weight(sample_channel_config: ChannelConfig) -> None:
    """Attachments should use configurable weight."""
    config = sample_channel_config.model_copy(update={"file_weight": 9.5})
    features = ScoringFeatures(
        has_files=True,
        channel_name="test",
    )

    score = scoring_engine.calculate_score(features, config)

    assert score == 9.5
    assert "attachments weight +9.5" in features.explanations


def test_calculate_score_trusted_bot_bypass(
    sample_channel_config: ChannelConfig,
) -> None:
    """Trusted bots should bypass the penalty with explanation."""
    config = sample_channel_config.model_copy(
        update={"trusted_bots": ["U_TRUSTED", "B987"]}
    )
    features = ScoringFeatures(
        is_bot=True,
        channel_name="test",
        author_id="U_TRUSTED",
    )

    score = scoring_engine.calculate_score(features, config)

    assert score == 0.0
    assert "trusted bot bypass" in features.explanations


def test_calculate_score_untrusted_bot_penalty(
    sample_channel_config: ChannelConfig,
) -> None:
    """Untrusted bots should still incur the penalty."""
    config = sample_channel_config.model_copy(update={"trusted_bots": ["U_TRUSTED"]})
    features = ScoringFeatures(
        is_bot=True,
        channel_name="test",
        author_id="U_OTHER",
    )

    score = scoring_engine.calculate_score(features, config)

    assert score == sample_channel_config.bot_penalty
    assert "bot penalty applied" in features.explanations


def test_calculate_score_anchors_capped(sample_channel_config: ChannelConfig) -> None:
    """Test anchor score is capped at 12."""
    features = ScoringFeatures(
        has_keywords=False,
        keyword_count=0,
        has_mention=False,
        reply_count=0,
        reaction_count=0,
        anchor_count=10,  # Would be 40 without cap
        link_count=0,
        has_files=False,
        is_bot=False,
        channel_name="test",
    )

    score = scoring_engine.calculate_score(features, sample_channel_config)

    # Capped at 12.0
    assert score == 12.0


def test_calculate_score_links_capped(sample_channel_config: ChannelConfig) -> None:
    """Test link score is capped at 6."""
    features = ScoringFeatures(
        has_keywords=False,
        keyword_count=0,
        has_mention=False,
        reply_count=0,
        reaction_count=0,
        anchor_count=0,
        link_count=10,  # Would be 20 without cap
        has_files=False,
        is_bot=False,
        channel_name="test",
    )

    score = scoring_engine.calculate_score(features, sample_channel_config)

    # Capped at 6.0
    assert score == 6.0


def test_score_message_integration(
    sample_slack_message: SlackMessage, sample_channel_config: ChannelConfig
) -> None:
    """Test complete message scoring."""
    score, features = scoring_engine.score_message(
        sample_slack_message, sample_channel_config
    )

    assert score > 0
    assert features.has_keywords is True
    assert features.anchor_count == 1
    assert features.reaction_count == 7


def test_is_candidate_above_threshold() -> None:
    """Test candidate selection above threshold."""
    assert scoring_engine.is_candidate(20.0, 15.0) is True


def test_is_candidate_below_threshold() -> None:
    """Test candidate selection below threshold."""
    assert scoring_engine.is_candidate(10.0, 15.0) is False


def test_is_candidate_exactly_threshold() -> None:
    """Test candidate selection exactly at threshold."""
    assert scoring_engine.is_candidate(15.0, 15.0) is True
